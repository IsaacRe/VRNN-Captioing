import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.distributions import Normal
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        return Variable(x, volatile=volatile).cuda()
    return Variable(x, volatile=volatile)

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
    
"""
Implementation of Variational RNN (LSTM)
"""
class VRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, latent_size, num_layers_lstm):
        """Set the hyper-parameters and build the layers."""
        super(VRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + latent_size, hidden_size, num_layers_lstm, batch_first=True)
        # q(z|x, h)
        self.q_z = nn.Linear(embed_size + hidden_size, latent_size * 2)
        # p(z|h)
        self.prior = nn.Linear(hidden_size, latent_size * 2)
        # gaussian noise generator for re-paramaterization trick
        self.normal = Normal(torch.zeros(latent_size,), torch.ones(latent_size,))
        # q(x|z) backwards inference
        self.q_x = nn.Linear(latent_size + hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.q_z.weight.data.uniform_(-0.1, 0.1)
        self.q_z.bias.data.fill_(0)
        self.prior.weight.data.uniform_(-0.1, 0.1)
        self.prior.bias.data.fill_(0)
        self.q_x.weight.data.uniform_(-0.1, 0.1)
        self.q_x.bias.data.fill_(0)

    """
    Conduct forward pass, outputting prior, posterior, and inference distributions
    """
    def forward(self, features, captions, lengths, states=None, z_0=None):
        """ Decode image feature vectors and generates captions. """
        
        z_padding = to_var(torch.zeros(features.shape[0], self.q_z.out_features / 2))
        if z_0 is None:
            z_0 = z_padding
        features = torch.cat([features, z_0], dim=1).unsqueeze(1)

        # conduct initial lstm step to embed image features in internal states
        h, states = self.lstm(features, states)
        h = h.squeeze(1)
        embeddings = self.embed(captions)

        p_mus, p_sigmas, q_mus, q_sigmas, q_xs = [], [], [], [], []
        for i in range(max(lengths)):
            # conduct forward pass for VRNN cell

            # get tuple of (mean, std dev) for prior from h_tm1
            p_mu, p_sigma = self.prior(h).chunk(2, dim=1)
            # get tuple of (mean, var) for q_z from x_t and h_tm1
            q_mu, q_sigma = self.q_z(torch.cat([embeddings[:,i], h], dim=1)).chunk(2, dim=1)
            # sample from q_z using reparameterization to get z - we take n=batch size samples
            z = to_var(self.normal.sample_n(q_mu.shape[0])) * q_sigma + q_mu
            # get q_x from z_t and h_tm1
            q_x = self.q_x(torch.cat([z, h], dim=1))
            q_x = nn.functional.log_softmax(q_x, dim=1)
            # perform lstm step to get h_t from x_t, z_t, h_tm1 - unsqueeze for lstm api
            inputs = torch.cat([embeddings[:,i], z], dim=1).unsqueeze(1)
            h, states = self.lstm(inputs, states)
            h = h.squeeze(1)

            p_mus.append(p_mu)
            p_sigmas.append(p_sigma)
            q_mus.append(q_mu)
            q_sigmas.append(q_sigma)
            q_xs.append(q_x)


        # pack padded sequence to mask out padding terms and flatten batch + step dimensions
        [p_mus, p_sigmas, q_mus, q_sigmas, q_xs] = [pack_padded_sequence(torch.stack(t, dim=1), lengths, batch_first=True)[0] for t in [p_mus, p_sigmas, q_mus, q_sigmas, q_xs]]

        return (p_mus, p_sigmas), (q_mus, q_sigmas), q_xs

    def sample(self, features, ground_truth, h_0=None, c_0=None):
        pass

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

    def introspective_forward(self, features, captions, lengths, hints = 0, states= None):
        outputs = []
        inputs = features.unsqueeze(1)
        for i in range(lengths[0]):                            
            hiddens, states = self.lstm(inputs, states)    # (batch_size, 1, hidden_size), 
            output = self.linear(hiddens.squeeze(1))      # (batch_size, vocab_size)
            outputs.append(output)
            predicted = output.max(1)[1]
            if hints > 0:
                inputs = self.embed(captions[:, i])         # if hints > 0, instead of passing previous predictions to generate the next, we pass in the ground truth instead.
                hints = hints - 1                           # decrement hints by 1.
            else:                                           # if hints == 0, we use the prediction.
                inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                    # (batch_size, 1, embed_size)
        outputs = torch.stack(outputs, 1)                     # stack the outputs to BxTx* shape.  
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True) # pack the shape with lengths
        return outputs[0]
    
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

     
    def update_cell_states(self,states,states_no_update,prev_h,user_input,inputs,input_size,vocab,c_step):
        # loop through all steps for which we have ground truth
        criterion = nn.CrossEntropyLoss()
        for i in range(input_size):
            if i > 0:
                prev_h = states[0]
                # conduct lstm step to get next cell state
                _, states = self.lstm(inputs, states)
                _, states_no_update = self.lstm(inputs, states_no_update)


            # create parameters we can update
            c_param,h_param = nn.Parameter(states[1].data),nn.Parameter(states[0].data)
            states = (h_param, c_param)
            temp_states = (states[0].clone(),states[1].clone())
            updates = []
            # conduct prop 1 update (update current step's cell state), append to updates
            # method creates hidden output from cell state (so gradient can backprop to c_param)
            h1 = self.h_from_c(inputs, prev_h, c_param)
            prediction = self.linear(h1.squeeze(0))
            # only update if prediction is incorrect
            if prediction.max(1)[1][0].data[0] != user_input[0,i].data[0]:
                loss = criterion(prediction, user_input[:,i])
                loss.backward(retain_graph=True)
                updates.append(torch.cuda.FloatTensor(c_param.grad.data.clone()))
                c_param.grad.data.zero_()
            # loop through all lstm steps (for which we have gt) at once to get predictions
            if input_size > i + 1:
                hiddens = []
                hiddens, _ = self.lstm(self.embed(user_input[:,i:-1]), states)
                hiddens = torch.split(hiddens.squeeze(0), 1, dim=0)
                num_hiddens = len(hiddens)
            else:
                num_hiddens = 0
            # Get update for each step of lstm
            for j in range(num_hiddens):
                # conduct prop i+1 update for each step executed
                prediction = self.linear(hiddens[j])
                if prediction.max(1)[1][0].data[0] != user_input[0,i+j+1].data[0]:
                    loss = criterion(prediction, user_input[:,i+j+1])
                    loss.backward(retain_graph=True)
                    updates.append(torch.cuda.FloatTensor(c_param.grad.data.clone()))
                    c_param.grad.data.zero_()

            # Proceed to do the actuall updates on current cell state
            for update in updates:
                states[1].data -= update * c_step
            states[0].data = self.h_from_c(inputs, prev_h, states[1]).data
        

            inputs = self.embed(user_input[:,i].unsqueeze(0))
        return states,states_no_update,inputs



    """
    Sample a sequence, updating with given c_step at each step. Backpropping from later steps
    to compute update of earlier step is currently non-functional - update is computed solely
    from cross entropy loss obtained from predictions computed during same step.
        Return : sampled ids, each step's predictions
        c_step : float specifying magnitude of each update
    """
    def sample_with_update(self, features, user_input, vocab, states=None, c_step=0.0, compare_steps=10, update_method='c', update_step=2, accum_updates=True):

        # Make user input a tensor
        user_input = Variable(user_input.unsqueeze(0).cuda())
        input_size = user_input.size(1) if user_input.size(1) > 1 else 0
        # inputs to the lstm must have 2 dimensions
        features = features.unsqueeze(1)
        # initialize prev_h
        if states == None:
            prev_h = Variable(torch.zeros(1,1,self.lstm.hidden_size).cuda())
        else:
            prev_h = states[0]
        # Get cell state output by first step
        inputs = features
        _, states = self.lstm(inputs, states)
        states_no_update = (states[0].clone(), states[1].clone())
        states,states_no_update,inputs = self.update_cell_states(states,states_no_update,prev_h,user_input,inputs,input_size,vocab,c_step)
        outputs = []
        sampled_ids = [torch.LongTensor([[0]] * 2)] * (input_size)
        inputs = torch.cat([inputs.clone(), inputs.clone()], 0)
        # Use final updated c_step to sample the remaining predictions
        states = (torch.cat([states_no_update[0], states[0]], 1),
                  torch.cat([states_no_update[1], states[1]], 1)) # states [0][1] should be in size 1, batchsize, hidden
        # print states
        for i in range(20 - input_size):
            hiddens, states = self.lstm(inputs, states)
            output = self.linear(hiddens.squeeze(1))
            outputs.append(output.data.cpu())
            
            predicted = output.max(1)[1]
            sampled_ids.append(predicted.data.cpu().view(2,1))
            inputs = self.embed(predicted).unsqueeze(1)

        all_predictions = torch.stack(outputs, 1)
        sampled_ids = torch.cat(sampled_ids, 1)

        return sampled_ids, all_predictions


    """
    Conduct gradient step on a particular step's cell state
        c : a single cell state tensor
    """
    def update_c(self,inputs,states,gt,c_step,prop_step):
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

    def sample(self, features, user_input,vocab, states=None, c_step=0.0,prop_step=1,update_method='c'):
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
                    states_history[-dist][1].data = self.update_c(inputs_history[-dist], states_history[-dist], ground_truth.squeeze(0), c_step,dist)
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
