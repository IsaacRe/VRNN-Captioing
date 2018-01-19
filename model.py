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

    """
    Conduct gradient step on a particular step's hidden output
        c : a single cell state tensor
    """
    def update_c(self, inputs, states, gt, c_step):

        c_param = nn.Parameter(states[1].data)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([c_param], lr=c_step)
        hiddens, _ = self.lstm(inputs,(states[0],c_param))
        predictions = self.linear(hiddens.squeeze(1))           
        loss = criterion(predictions, gt)
        
        loss.backward(retain_graph=True)
        optimizer.step()

        return c_param.data

    def sample(self, features, user_input,vocab, states=None, c_step=0.0):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        predictions = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            previous_state = states
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1)) # (batch_size, vocab_size)
            #print(str(torch.max(outputs.data))+'\n'+str(torch.min(outputs.data))+'\n'+str(torch.mean(outputs.data)))
            predicted = outputs.max(1)[1].unsqueeze(0)
            if i < len(user_input):
                ground_truth = Variable(torch.cuda.LongTensor([[user_input[i]]]))
                if c_step > 0 and predicted.data[0][0] != ground_truth.data[0][0]:
                    #print vocab.idx2word[predicted.data[0][0]]
                    #print vocab.idx2word[ground_truth.data[0][0]]
                    #print "backward"
                    previous_state[1].data = self.update_c(inputs, previous_state, ground_truth.squeeze(0), c_step)
                    hiddens,states = self.lstm(inputs,previous_state)
                    outputs = self.linear(hiddens.squeeze(1)) 
                    predicted = outputs.max(1)[1].unsqueeze(0)
                predicted = ground_truth
            sampled_ids.append(predicted)
            predictions.append(outputs.data.cpu())
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                 # (batch_size, 20)
        predictions = torch.stack(predictions, 1)
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
