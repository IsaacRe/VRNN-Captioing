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
        embedding_pred = features.unsqueeze(1)
        state = None
        
        out_1 = []
        out_0 = []
        for i in range(embeddings.size(1)):
            out, _ = self.lstm(embedding_pred, state)
            out_0.append(out)
            out, state = self.lstm(embeddings[:,i:i+1], state)
            out_1.append(out)
            embedding_pred = self.embed(self.linear(out.squeeze(1)).max(1)[1].unsqueeze(1))
        out_1 = torch.cat(out_1, 1)
        out_0 = torch.cat(out_0, 1)
        out_1 = pack_padded_sequence(out_1, lengths, batch_first=True)[0]
        out_0 = pack_padded_sequence(out_0, lengths, batch_first=True)[0]
        assert out_0.size() == out_1.size()
        return out_0, out_1

    def test(self, features, captions, length):
        """Takes in single sample (batch_size = 1)"""
        """Samples captions for given input features, returning a predicted sequence for each sequence of ground truths"""
        sequences = []
        lin_outs = torch.zeros(length - 1, 2, 1, 9956) # (length-1, 2, 1, vocab_size)
        next_input = features.unsqueeze(1)
        out_states = None
        """Loop over each ground truth to add to the sequence"""
        for j in range(length):
            hiddens, out_states = self.lstm(next_input, out_states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            if len(lin_outs) > 0:
                lin_outs[j-1, 1] = outputs.data
            sampled_ids = [predicted]
            next_input = self.embed(captions[:,j]).unsqueeze(1)
            inputs, states = self.embed(predicted).unsqueeze(1), out_states
            """Loop over each additional word in sequence to predict"""
            for i in range(length-j):     # maximum sampling length
                hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
                outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
                predicted = outputs.max(1)[1]
                if i == 0 and j < length - 1:
                    lin_outs[j, 0] = outputs.data
                sampled_ids.append(predicted)
                inputs = self.embed(predicted).unsqueeze(1)
            sampled_ids = torch.stack(sampled_ids, 1)                  # (batch_size, 20)
            sequences.append(sampled_ids.squeeze())
        return sequences, Variable(lin_outs, volatile=True)


    def sample(self, features,user_input,states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))  #hiddens.squeeze(1)          # (batch_size, vocab_size)
            predicted = outputs.max(1)[1].unsqueeze(0)
            if i < len(user_input):
                predicted = Variable(torch.cuda.LongTensor([[user_input[i]]]))
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()

    def next_word(self, features, user_input, word_number,states=None):
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
