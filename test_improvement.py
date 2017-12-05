import argparse
import nltk
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, 1,
                             shuffle=True, num_workers=args.num_workers) 

    # Find pretrained models to use
    filenames_split = [filename.split('-') for filename in os.listdir(args.model_path)]
    encoder_states = [f[1] + f[2][0] for f in filenames_split if 'encoder' in f[0]]
    decoder_states = [f[1] + f[2][0] for f in filenames_split if 'decoder' in f[0]]
    if encoder_states == [] or decoder_states == []:
        print("No models found in {} .".format(args.model_path))
        return
    encoder_max = str(np.max(np.array(encoder_states,dtype=int)))
    decoder_max = str(np.max(np.array(decoder_states,dtype=int)))

    encoder_state = 'encoder-{}-{}000.pkl'.format(encoder_max[0],encoder_max[1])
    decoder_state = 'decoder-{}-{}000.pkl'.format(decoder_max[0],decoder_max[1])

    print("Using encoder: {}".format(encoder_state))
    print("Using decoder: {}".format(decoder_state))

    # Build the models
    encoder = EncoderCNN(args.embed_size)
    encoder.eval() # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.model_path + encoder_state))
    decoder.load_state_dict(torch.load(args.model_path + decoder_state))

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    criterion = nn.CrossEntropyLoss()

    # Test
    for i, (images, captions, lengths) in enumerate(data_loader):
        
        if i > args.num_iters:
            break
        
        # Set mini-batch dataset
        images = to_var(images, volatile=True)
        print captions
        captions = to_var(captions, volatile=True)
        
        # Iterate through sequence
        losses = []
        bleu_scores = []
        partial_bleu_scores = []
        
        features = encoder(images)
        
        sequences, lin_outs = decoder.test(features, captions, lengths[0])
        lin_outs = lin_outs.cuda()
        # Calculate loss for current predicted item in sequence
        for j in range(lengths[0]):
            if j < lengths[0] - 1:
                loss = (criterion(lin_outs[j,0], captions[:,j]), criterion(lin_outs[j,1], captions[:,j]))
                losses.append(loss)

            # Get predicted and ground truth sentences
            partial_sentence = [vocab.idx2word[w.data[0]] for w in sequences[j]]
            full_sentence = ' '.join(partial_sentence)
            if j > 0:
                full_sentence = ' '.join([vocab.idx2word[w.data[0]] for w in captions[0,:j]]) + ' ' + full_sentence
            full_sentence = full_sentence.split()
            partial_gt = [vocab.idx2word[w.data[0]] for w in captions[0,j:]]
            full_gt = [vocab.idx2word[w.data[0]] for w in captions[0]]

            bleu_score = nltk.translate.bleu_score.sentence_bleu([full_gt],full_sentence)
            partial_bleu_score = nltk.translate.bleu_score.sentence_bleu([partial_gt],partial_sentence)
            bleu_scores.append(bleu_score)
            partial_bleu_scores.append(partial_bleu_score)

        print("Iteration {}:".format(i))
        print("Ground truth sentence: {}".format(full_gt))

        print("Comparing loss:")
        for l in losses:
            print("{} {}".format(l[0].data[0],l[1].data[0]))
           
        print("Comparing bleu scores:")
        for b in bleu_scores:
            print(b)

        print("Comparing partial bleu scores:")
        for b in partial_bleu_scores:
            print(b)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/resized2014' ,
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    parser.add_argument('--num_iters', type=int , default=1000,
                        help='number of samples on which to test performance')

    # Model parameters
    parser.add_argument('--theta', type=float , default=0.4,
                        help='value of theta used in calculating difference in losses')
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
