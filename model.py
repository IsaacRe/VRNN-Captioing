import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """ Decode image feature vectors and generates captions. """
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def h_from_c(self, x, h, c):
        # get weights
        w_io = Variable(self.lstm.weight_ih_l0.data[self.lstm.hidden_size*3:])
        w_ho = Variable(self.lstm.weight_hh_l0.data[self.lstm.hidden_size*3:])
        b_io = Variable(self.lstm.bias_ih_l0.data[self.lstm.hidden_size*3:].unsqueeze(1))
        b_ho = Variable(self.lstm.bias_hh_l0.data[self.lstm.hidden_size*3:].unsqueeze(1))

        # transpose
        h = h.view(h.size(2), -1)
        x = x.view(x.size(2), -1)
        c = c.view(c.size(2), -1)

        """
        # debug
        print("x : " + str(x.size()))
        print("h : " + str(h.size()))
        print("c : " + str(c.size()))
        print("w_io : " + str(w_io.size()))
        print("w_ho : " + str(w_ho.size()))
        print("b_io : " + str(b_io.size()))
        print("b_ho : " + str(b_ho.size()))
        """

        # output gate
        o = nn.functional.sigmoid(torch.mm(w_io, x) + torch.mm(w_ho, h) + b_io + b_ho)

        #print("o : " + str(o.size()))

        h1 = o * nn.functional.tanh(c)
        return h1.view(1,1,-1)

    def test_h_from_c(self):
        x = Variable(torch.normal(means=torch.ones(self.lstm.input_size)).view(1,1,-1)).cuda()
        h = Variable(torch.normal(means=torch.ones(self.lstm.hidden_size)).view(1,1,-1)).cuda()
        c = Variable(torch.normal(means=torch.ones(self.lstm.hidden_size)).view(1,1,-1)).cuda()
        

        hiddens, states = self.lstm(x, (h, c))

        h1 = states[0]

        # Calculate hidden from new cell state
        h2 = self.h_from_c(x, h, states[1])

        print(h1 - h2)
        print(torch.mean(h1 - h2))
        


    """
    Conduct gradient step on a particular step's hidden output
        c : a single cell state tensor
    """
    def update_c(self, inputs, h, c, gt, c_step):

        c_param = nn.Parameter(c.data)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([c_param], lr=c_step)
        h1 = self.h_from_c(inputs, h, c_param)
        predictions = self.linear(h1.squeeze(0))           
        loss = criterion(predictions, gt)
        
        loss.backward(retain_graph=True)
        optimizer.step()

        # get new hidden output from updated cell state
        h1 = self.h_from_c(inputs, h, c_param)

        return h1.data, c_param.data

    def sample_with_update(self, features, user_input, vocab, states=None, c_step=0.0):
        user_input = Variable(torch.cuda.LongTensor([user_input]))
        features = features.unsqueeze(1)
        
        criterion = nn.CrossEntropyLosss()

        prev_h = states[0]
        # Get cell state output by first step
        inputs = features
        _, states = self.lstm(inputs, states)
        state_hist = []

        # loop through all steps for which we have ground truth
        for i in range(len(user_input)):
            if i > 0:
                prev_h = states[0]
                # conduct lstm step to get next cell state
                _, states = self.lstm(inputs, states)
            c_param = nn.Parameter(states[1].data)
            
            updates = []

            # conduct prop 0 update, append to updates
            h1 = self.h_from_c(inputs, prev_h, c_param)
            prediction = self.linear(h1.squeeze(0))
            loss = criterion(prediction, user_input[:,i])
            loss.backward(retain_graph=True)
            updates.append(c_param.grad.data)
            c_param.grad.data.zero_()

            # loop through all lstm steps (for which we have gt) at once to get predictions
            hiddens, _ = self.lstm(user_input[:,i:-1], states)
            hiddens = hiddens.squeeze(0)

            if len(user_input) - 1 == i:
                assert hiddens.size(0) == 0

            # Get update for each step of lstm
            for j in range(hiddens.size(0)):
                # conduct prop i+1 update for each step executed
                prediction = self.linear(hiddens[j]).max(1)[1]
                loss = criterion(prediction, user_input[:,i+j+1])

                loss.backward(retain_graph=True)

                updates.append(c_param.grad.data)
                c_param.grad.data.zero_()

            # apply updates
            states[1].data += torch.sum(torch.stack(updates, dim=0), dim=0) * c_step

            inputs = user_input[i]

        all_predictions = []
        sampled_ids = []
        # Use final updated c_step to sample the remaining predictions
        for i in range(20 - len(user_input)):
            hiddens, states = self.lstm(inputs, states)
            predictions = self.linear(hiddens.squeeze(1))
            all_predictions.append(predictions)
            
            predicted = predictions.max(1)[1].unsqueeze(1)
            sampled_ids.append(predicted)

            inputs = self.embed(predicted)

        all_predictions = torch.stack(all_predictions, 1)
        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze(), all_predictions.squeeze()

    def sample(self, features, user_input,vocab, states=None, c_step=0.0):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        predictions = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            previous_hidden = Variable(torch.zeros(1, 1, self.lstm.hidden_size)).cuda() \
                    if states == None else states[0]
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1)) # (batch_size, vocab_size)
            predicted = outputs.max(1)[1].unsqueeze(0)
            if i < len(user_input):
                ground_truth = Variable(torch.cuda.LongTensor([[user_input[i]]]))
                if c_step > 0 and predicted.data[0][0] != ground_truth.data[0][0]:
                    states[0].data, states[1].data = self.update_c(inputs, previous_hidden, states[1], ground_truth.squeeze(0), c_step)
                predicted = ground_truth
            sampled_ids.append(predicted)
            predictions.append(outputs.data.cpu())
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                 # (batch_size, 20)
        predictions = torch.stack(predictions, 1)
        return sampled_ids.squeeze(), predictions.squeeze()

    def update_c_beta(self,inputs,states,gt,c_step,prop_step):
        c_param = nn.Parameter(states[1].data)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([c_param], lr=c_step)
        states = (states[0],c_param)
        outputs = None

        for i in range(prop_step):
            hiddens, states = self.lstm(inputs,states)
            outputs = self.linear(hiddens.squeeze(1))
            predictions = outputs.max(1)[1].unsqueeze(0)         
            inputs = self.embed(predictions)

        loss = criterion(outputs, gt)
        
        loss.backward(retain_graph=True)
        optimizer.step()

        return c_param.data

    def sample_beta(self, features, user_input,vocab, states=None, c_step=0.0,prop_step=1):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        predictions = []
        inputs = features.unsqueeze(1)
        states_history = [states]
        inputs_history = []
        for i in range(20):                                      # maximum sampling length
            inputs_history.append(inputs)
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            states_history.append(states)
            outputs = self.linear(hiddens.squeeze(1)) # (batch_size, vocab_size)
            predicted = outputs.max(1)[1].unsqueeze(0)
            if i < len(user_input):
                ground_truth = Variable(torch.cuda.LongTensor([[user_input[i]]]))
                if c_step > 0 and predicted.data[0][0] != ground_truth.data[0][0]:
                    
                    dist = i if prop_step>i else prop_step
                    states_history[-dist][1].data = self.update_c_beta(inputs_history[-dist], states_history[-dist], ground_truth.squeeze(0), c_step,dist)
                    inputs = inputs_history[-dist]
                    states = states_history[-dist]
                    for p in range(dist):
                        hiddens, states = self.lstm(inputs,states)
                        outputs = self.linear(hiddens.squeeze(1))
                        predicted = outputs.max(1)[1].unsqueeze(0)         
                        # predicted = Variable(torch.cuda.LongTensor([[user_input[-dist+p]]]))
                        inputs = self.embed(predicted)

                predicted = ground_truth
            sampled_ids.append(predicted)
            predictions.append(outputs.data.cpu())
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                 # (batch_size, 20)
        predictions = torch.stack(predictions, 1)
        # print "sampled id"
        # print sampled_ids.squeeze()
        # print "predictions"
        # print predictions.squeeze()
        return sampled_ids.squeeze(), predictions.squeeze()


    def next_word(self, features, user_input, word_number, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            if i < len(user_input):
                predicted = Variable(torch.cuda.LongTensor([[user_input[i]]]))
                inputs = self.embed(predicted)
            else:
                st= outputs.sort(1,descending=True)
                for a in range(0,word_number):
                    sampled_ids.append(st[1][0][a])
                break
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        return sampled_ids.squeeze()       
