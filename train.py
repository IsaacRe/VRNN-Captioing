import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import sys
from utils.data_loader import get_loader 
from utils.build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from datetime import datetime

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    
    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    val_loader = get_loader('./data/val_resized2014/', './data/annotations/captions_val2014.json',
                             vocab, transform, 1, False, 1)

    start_epoch = 0

    encoder_state = args.encoder
    decoder_state = args.decoder
    
    # Build the models
    encoder = EncoderCNN(args.embed_size)
    if not args.train_encoder:
        encoder.eval()
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    if args.restart:
        encoder_state, decoder_state = 'new', 'new'

    if encoder_state == '': encoder_state = 'new'
    if decoder_state == '': decoder_state = 'new'

    if decoder_state != 'new':
        start_epoch = int(decoder_state.split('-')[1])

    print("Using encoder: {}".format(encoder_state))
    print("Using decoder: {}".format(decoder_state))

        
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    """ Make logfile and log output """
    with open(args.model_path + args.logfile, 'a+') as f:
        f.write("Training on vanilla loss (using new model). Started {} .\n".format(str(datetime.now())))
        f.write("Using encoder: new\nUsing decoder: new\n\n")
    
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    batch_loss = []
    batch_acc = []
    
    # Train the Models
    total_step = len(data_loader)
    for epoch in range(start_epoch, args.num_epochs):
        for i, (images, captions, lengths,_,_) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            out = decoder(features, captions, lengths)
            loss = criterion(out, targets)
            batch_loss.append(loss.data[0])
            
            loss.backward()
            optimizer.step()


            # # Print log info
            # if i % args.log_step == 0:
            #     print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f, Val: %.5f, %.5f'
            #           %(epoch, args.num_epochs, i, total_step, 
            #             loss.data[0], np.exp(loss.data[0]), acc, gt_acc)) 
                
            #     with open(args.model_path + args.logfile, 'a') as f:
            #         f.write('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f, Val: %.5f, %.5f\n'
            #               %(epoch, args.num_epochs, i, total_step, 
            #                 loss.data[0], np.exp(loss.data[0]), acc, gt_acc)) 
                
            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                with open(args.model_path + 'training_loss.pkl', 'w+') as f:
                    pickle.dump(batch_loss, f)
                with open(args.model_path + 'training_val.pkl', 'w+') as f:
                    pickle.dump(batch_acc, f)
    with open(args.model_path + args.logfile, 'a') as f:
        f.write("Training finished at {} .\n\n".format(str(datetime.now())))
                
                
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
    parser.add_argument('--logfile', type=str , default='logfile',
                        help='specify the logfile')
    
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
    parser.add_argument('--encoder', type=str, default='models/encoder_pretrained.pkl')
    parser.add_argument('--decoder', type=str, default='')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--train_encoder', action='store_true')
    parser.add_argument('--val_step', type=int, default=100)
    args = parser.parse_args()
    print(args)
    main(args)
