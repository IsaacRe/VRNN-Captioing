import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model_new import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from datetime import datetime
from scipy.special import expit as sig

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

        
    # Load the trained model parameters
    if encoder_state != 'new':
        encoder.load_state_dict(torch.load(encoder_state))
    if decoder_state != 'new':
        decoder.load_state_dict(torch.load(decoder_state))
    
    """ Make logfile and log output """
    with open(args.model_path + args.logfile, 'a+') as f:
        f.write("Training on gated loss (log difference: {}). Started {} .\n".format(args.log, str(datetime.now())))
        f.write("Using encoder: {}\nUsing decoder: {}\n\n".format(encoder_state, decoder_state))
    
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(reduce=False)
    gate = nn.ReLU()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)
    alpha = args.initial_a
    delta_a = args.delta_a

    alpha_updates = []
    batch_loss = []
    for epoch in range(start_epoch, args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            print("Training with gradient gating (log difference: {}). Current Epoch: {}".format(args.log, epoch))

            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            # Skip first word in each sequence
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, Backward and Optimize
            decoder.zero_grad()
            if not args.train_encoder:
                features = encoder(images).detach()
                print('Only Training Decoder!!!!!!!')
            else:
                features = encoder(images)
            out_0, out = decoder(features, captions, lengths)
            losses = criterion(out, targets)
            losses_0 = criterion(out_0.detach(), targets)

            mask = np.where((losses.cpu().data.numpy()*(1+alpha) - losses_0.cpu().data.numpy()) > 0)[0]
            # If no samples pass
            if mask.shape[0] == 0:
                message = 'Alpha: {}\nDelta_A: {}\nNo samples passed gradient gate!!!!\n \
                        Updating alpha and skipping...'.format(alpha, delta_a)
                print(message)
                with open(args.model_path + args.logfile, 'a') as f:
                    f.write(message + '\n')
                alpha += delta_a
                alpha_updates.append(delta_a)
                continue                

            mask = torch.LongTensor(mask) # Mask for the losses we will keep
            loss = torch.sum(losses[mask.cuda()]) / mask.size(0)
            
            # Calculate ratio of samples passing loss
            ratio = float(mask.size(0)) / losses.size(0)
            print("Ratio of samples passing gradient: {}".format(ratio))
            print("Current Alpha: {}".format(alpha))
            print("Current Delta_A: {}".format(delta_a))

            # Update alpha
            if ratio < args.pass_ratio - args.pass_margin:
                alpha += delta_a
                alpha_updates.append(delta_a)
            elif ratio > args.pass_ratio + args.pass_margin and alpha > delta_a:
                alpha -= delta_a
                alpha_updates.append(-delta_a)
            else:
                # Reset if we didn't update alpha this iteration
                alpha_updates = []

            # Choose how to change delta_a
            if len(alpha_updates) > args.a_log_length:
                alpha_updates.remove(alpha_updates[0]) # Only look at last 10 updates
            if np.abs(np.sum(alpha_updates)) < delta_a and len(alpha_updates) == args.a_log_length:
                if delta_a / 2 > 0:
                    delta_a /= 2 # If we can't get enough precision with current delta_a, make it smaller
                alpha_updates = []
            elif np.abs(np.sum(alpha_updates)) > delta_a*(args.a_log_length - 1):
                delta_a *= 2 # If we can't update fast enough, make delta_a larger
                alpha_updates = []
                
            
            print("Max loss: {}".format(torch.max(losses)))
            print("Max loss0: {}".format(torch.max(losses_0)))

            print("Min loss: {}".format(torch.min(losses)))
            print("Min loss0: {}".format(torch.min(losses_0)))

            print loss
            batch_loss.append(loss.data[0])

            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f, Alpha: %.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]), sig(alpha))) 
                
                with open(args.model_path + args.logfile, 'a') as f:
                    f.write('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f, \
                            Alpha: %.4f, Ratio: %.2f\n'
                          %(epoch, args.num_epochs, i, total_step, 
                            loss.data[0], np.exp(loss.data[0]), sig(alpha), ratio)) 

            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                with open(args.model_path + 'training_loss.pkl', 'wb+') as f:
                    pickle.dump(batch_loss, f)

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
    parser.add_argument('--restart', action='store_true',
                        help='whether to restart the epoch-batch counter \
                                for the training process. If False, training \
                                will cap at num-epochs.')
    parser.add_argument('--encoder', type=str , default='./models/encoder_pretrained.pkl',
                        help='specify the encoder parameter file to begin training \
                                from. If not specified, will choose most recent.')
    parser.add_argument('--decoder', type=str , default='',
                        help='specify the decoder parameter file to begin training \
                                from. If not specified, will choose most recent.')
    parser.add_argument('--logfile', type=str , default='train.log',
                        help='specify logfile')

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
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--delta_a', type=float, default=1e-5)
    parser.add_argument('--log', action='store_true',
                        help='whether to gate based on log difference')
    parser.add_argument('--initial_a', type=float, default=1e-5,
                        help='specify desired initial value of alpha')
    parser.add_argument('--pass_ratio', type=float, default=0.5,
                        help='specify ratio of samples that should pass loss \
                                through gate in each batch. When observed ratio is \
                                greater than the given value, alpha will be updated.')
    parser.add_argument('--pass_margin', type=float, default=0.05,
                        help='specify far ratio can diverge from specified pass_ratio \
                                before alpha is updated')
    parser.add_argument('--train_encoder', action='store_true')
    parser.add_argument('--a_log_length', type=int, default=6)
    args = parser.parse_args()
    print(args)
    main(args)
