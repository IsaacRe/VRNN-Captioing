import argparse
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
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    start_epoch = 0

    if not args.restart:
        # Find pretrained models to use
        filenames_split = [filename.split('-') for filename in os.listdir(args.model_path)]

        if args.encoder == '':
            encoder_states = [f[1] + f[2][0] for f in filenames_split if 'encoder' in f[0]]
            if encoder_states == []:
                print("No encoder models found in {} .".format(args.model_path))
                return
            encoder_max = str(np.max(np.array(encoder_states,dtype=int)))
            encoder_state = 'encoder-{}-{}000.pkl'.format(encoder_max[0],encoder_max[1])
        else:
            encoder_state = args.encoder

        if args.decoder == '':
            decoder_states = [f[1] + f[2][0] for f in filenames_split if 'decoder' in f[0]]
            if decoder_states == []:
                print("No decoder models found in {} .".format(args.model_path))
                return
            decoder_max = str(np.max(np.array(decoder_states,dtype=int)))
            decoder_state = 'decoder-{}-{}000.pkl'.format(decoder_max[0],decoder_max[1])
        else:
            decoder_state = args.decoder
            
        start_epoch = int(decoder_state.split('-')[1])

        print("Using encoder: {}".format(encoder_state))
        print("Using decoder: {}".format(decoder_state))

        # Build the models
        encoder = EncoderCNN(args.embed_size)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                             len(vocab), args.num_layers)
        
        # Load the trained model parameters
        encoder.load_state_dict(torch.load(args.model_path + encoder_state))
        decoder.load_state_dict(torch.load(args.model_path + decoder_state))

    else:
        # Build the models
        encoder = EncoderCNN(args.embed_size)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)

    for epoch in range(start_epoch, args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            print captions
            captions = to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            _, out = decoder(features, captions, lengths)
            loss = criterion(out, targets)
            print loss
            
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
                
            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                
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
    parser.add_argument('--restart', type=bool , default=True,
                        help='whether to restart the epoch-batch counter \
                                for the training process. If False, training \
                                will cap at num-epochs.')
    parser.add_argument('--encoder', type=str , default='',
                        help='specify the encoder parameter file to begin training \
                                from. If not specified, will choose most recent.')
    parser.add_argument('--decoder', type=str , default='',
                        help='specify the decoder parameter file to begin training \
                                from. If not specified, will choose most recent.')


    # Model parameters
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
