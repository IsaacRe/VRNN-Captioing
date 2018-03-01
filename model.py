import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.autograd import Variable
import pdb


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

    """
    (1) Sample a sequence, updating with given c_step at each step. Backpropping from later steps
        to compute update of earlier step is currently non-functional - update is computed solely
        from cross entropy loss obtained from predictions computed during same step.
        Return : sampled ids, each step's predictions
    (2) Sample a sequence, updating cell state of a specified step with each c_step value in the
        specified list
        Return : (len(c_steps) X compare_steps) tensor containing cross entropy losses,
                 (len(c_steps) X compare_steps) tensor containing raw ground truth prediction value
    c_step : (1) float specifying magnitude of each update
             (2) list(float) specifying c_step values to test when updating the last step for which
                 we have ground truth
    """
    def sample_with_update(self, features, user_input, vocab, states=None, c_step=0.0, compare_steps=10, update_method='c', update_step=2, accum_updates=True):
        input_size = 0
	if type(user_input) == list:
            if user_input != []:
                user_input = Variable(torch.cuda.LongTensor([user_input]))
                input_size = user_input.size(1)
            else:
                input_size = 0
        else:
            user_input = Variable(user_input.unsqueeze(0).cuda())
        features = features.unsqueeze(1)

        if type(c_step) == list:
            input_size = update_step

            if input_size > user_input.size(1):
                return None, None
        
        criterion = nn.CrossEntropyLoss()

        if states == None:
            prev_h = Variable(torch.zeros(1,1,self.lstm.hidden_size).cuda())
        else:
            prev_h = states[0]
        # Get cell state output by first step
        inputs = features

        _, states = self.lstm(inputs, states)
        state_hist = []

        ce_tensor = []
        pred_tensor = []

        # loop through all steps for which we have ground truth
        for i in range(input_size):
            if i > 0:
                prev_h = states[0]
                # conduct lstm step to get next cell state
                _, states = self.lstm(inputs, states)
            c_param = nn.Parameter(states[1].data)
            h_param = nn.Parameter(states[0].data)
            states = (h_param, c_param)
            
            updates = []

            # we will store cross entropy for predictions gen'd from states updated with each c_step
            #   (when testing for optimal c_step)
            ce_each = []
            pred_each = []

            # conduct prop 0 update, append to updates
            if update_method == 'c':
                h1 = self.h_from_c(inputs, prev_h, c_param)
                assert h1[0,0,0].data[0] - states[0][0,0,0].data[0] < 1e-4
                prediction = self.linear(h1.squeeze(0))
            else:
                prediction = self.linear(h_param.squeeze(0))

            if prediction.max(1)[1][0].data[0] != user_input[0,i].data[0] or type(c_step) == list:
                loss = criterion(prediction, user_input[:,i])
                
                # store for later comparison with predictions gen'd from various c_step updates
                ce_each.append(loss.data.cpu().clone())
                assert prediction.size(1) > 6000
                pred_each.append(prediction[:,int(user_input[0,i].data.cpu())].data.cpu().clone())
                
                loss.backward(retain_graph=True)
                
                if update_method == 'c':
                    
                    updates.append(torch.cuda.FloatTensor(c_param.grad.data.clone()))
                    c_param.grad.data.zero_()

                elif update_method == 'h':
                    updates.append(torch.cuda.FloatTensor(h_param.grad.data.clone()))
                    h_param.grad.data.zero_()


            # loop through all lstm steps (for which we have gt) at once to get predictions
            if input_size > i + 1:
                hiddens = []
                _states = states
                
                hiddens, _ = self.lstm(self.embed(user_input[:,i:-1]), states)
                hiddens = torch.split(hiddens.squeeze(0), 1, dim=0)

                num_hiddens = len(hiddens)

            else:
                num_hiddens = 0

            if user_input.size(1) - 1 == i:
                assert num_hiddens == 0

            if accum_updates:
                # Get update for each step of lstm
                for j in range(num_hiddens):
                    # conduct prop i+1 update for each step executed
                    prediction = self.linear(hiddens[j])

                    if prediction.max(1)[1][0].data[0] != user_input[0,i+j+1].data[0]:
                        loss = criterion(prediction, user_input[:,i+j+1])

                        loss.backward(retain_graph=True)
                        
                        if update_method == 'c':

                            assert c_param.grad.data[0,0][0] != 0.0

                            updates.append(torch.cuda.FloatTensor(c_param.grad.data.clone()))
                            c_param.grad.data.zero_()

                        elif update_method == 'h':

                            assert h_param.grad.data[0][0] != 0.0

                            updates.append(torch.cuda.FloatTensor(h_param.grad.data.clone()))
                            h_param.grad.data.zero_()
                
            else:
                assert len(updates) < 2

            # apply updates
            if updates != []:
                
                # if we're collecting data to test optimal c_step:
                if type(c_step) == list:

                    if i == input_size - 1:
                        # test predictions for all c_step values
                        test_states = [states]
                        for step in c_step:
                            if update_method == 'c':
                                updated_c = Variable(states[1].data.clone() + updates[0] * step)

                                # get new hidden output
                                new_h = self.h_from_c(inputs, prev_h, updated_c)

                                # appended states will be used to compute accuracy metrics of later steps
                                test_states.append((new_h, updated_c))

                            elif update_method == 'h':
                                new_h = Variable(states[0].data.clone() + updates[0] * step)
                                test_states.append((new_h, states[1]))

                            # compute new predictions
                            predictions = self.linear(new_h.squeeze(0))

                            # determine optimal c_step (using cross entropy loss as metric)
                            loss = criterion(predictions, user_input[:,i]).data.cpu()
                            pred = predictions[:,int(user_input[0,i].data.cpu())].data.cpu().clone()
                            ce_each.append(loss)
                            pred_each.append(pred)
                        
                        # add first row of output tensors
                        ce_tensor.append(torch.stack(ce_each, 0))
                        pred_tensor.append(torch.cat(pred_each, 0))
                    
                else:

                    if len(updates) > 1:
                        # calculate unit vectors for each update
                        assert updates[0].size(2) == self.lstm.hidden_size
                        units = [u/u.norm(2.0, dim=2) for u in updates]
                        units = torch.cat(units, dim=0).squeeze()
                        
                        # compute sum of dot-product with each other update unit vector
                        #  for each unit vector.
                        dots = torch.mm(units, units.transpose(0,1))
                        weights = torch.sum(dots, dim=0)
                        
                        assert weights.size(0) == len(updates)
                        update = torch.sum(torch.stack([updates[j]*weights[j] for j in range(len(updates))], dim=0), dim=0)

                    else:
                        update = updates[0]
                    

                    """
                    dots = [torch.stack([units[j] * units[k] for k in range(units.size(0)) if k != i], dim=0).norm(dim=0) for i in range(units.size(0))]

                    # weight updates by magnitude of dot product with all previous updates
                    idxs = sorted(range(len(dots)), key=lambda k: dots[k])
                    print(idxs)
                    """


                    

                    if update_method == 'c':
                        states[1].data -= update * c_step
                        states[0].data = self.h_from_c(inputs, prev_h, states[1]).data
                    elif update_method == 'h':
                        states[0].data -= update * c_step

            inputs = self.embed(user_input[:,i].unsqueeze(0))

        all_predictions = [torch.FloatTensor([[0] * self.linear.out_features])] * (input_size)
        sampled_ids = [torch.LongTensor([[0]])] * (input_size)
        
        """
        (2)
        """
        if type(c_step) == list:
            
            assert len(user_input.size()) == 2

            inputs = torch.cat([inputs]*(len(c_step) + 1), 0)
            states = (torch.cat([t[0] for t in test_states], 1), \
                      torch.cat([t[1] for t in test_states], 1))

            for i in range(compare_steps):
                if i + input_size >= user_input.size(1):
                    break

                hiddens, states = self.lstm(inputs, states)     # 1 X len(c_step) X hidden size
                predictions = self.linear(hiddens.squeeze(0))   # len(c_step) X vocab size
                
                loss = [criterion(predictions[j], user_input[0,i+input_size]).data.cpu() \
                            for j in range(predictions.size(0))]
                loss = torch.cat(loss, 0)
                assert loss.size(0) == len(c_step) + 1

                pred = predictions[:,0,user_input[0,i+input_size].data[0]].data.cpu().clone()
                ce_tensor.append(loss)
                pred_tensor.append(pred)

            return torch.stack(ce_tensor, 1).squeeze(), torch.stack(pred_tensor, 1)

        """
        (1)
        """
        # Use final updated c_step to sample the remaining predictions
        for i in range(20 - input_size):
            hiddens, states = self.lstm(inputs, states)
            predictions = self.linear(hiddens.squeeze(1))
            all_predictions.append(predictions.data.cpu())
            
            predicted = predictions.max(1)[1].unsqueeze(1)
            sampled_ids.append(predicted.data.cpu())

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

    def sample_beta(self, features, user_input,vocab, states=None, c_step=0.0,prop_step=1,update_method='c'):
        if prop_step == -1:
            return self.sample_with_update(features, user_input, vocab, states, c_step, update_method=update_method)
        elif prop_step == 1:
            return self.sample_with_update(features, user_input, vocab, states, c_step, update_method=update_method, accum_updates=False)

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
        sampled_ids = torch.cat(sampled_ids, 1).data.cpu()       # (batch_size, 20)
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
