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
import test_func

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
    batch_acc = []
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
            og_size = losses.size(0)
            losses_0 = criterion(out_0.detach(), targets)

            # Mask out samples with very low loss
            mask = np.minimum(np.where(losses.cpu().data.numpy() > args.min_loss)[0] + 1, losses.size(0)-1) # We mask by the loss of the preceeding index to determine if predicted word used as input is far enough from ground truth to expect an improvement
            # If no samples pass
            if mask.shape[0] == 0:
                message = 'Alpha: {}\nDelta_A: {}\nNo samples passed gradient gate!!!!\n \
                        Updating alpha and skipping...'.format(alpha, delta_a)
                print(message)
                with open(args.model_path + args.logfile, 'a') as f:
                    f.write(message + '\n')
                continue                

            mask = torch.LongTensor(mask) # Mask for samples with room for improvement
            losses, losses_0 = losses[mask.cuda()], losses_0[mask.cuda()]
            
            print("Ratio passing loss threshold: {}".format(float(losses.size(0)) / og_size))

            loss_diff = (losses / losses_0).cpu().data.numpy() if args.divide else (losses - losses_0).cpu().data.numpy()
            med_loss_diff = np.median(loss_diff)

            #mask = np.where((losses.cpu().data.numpy()*(1+alpha) - losses_0.cpu().data.numpy()) > 0)[0]
            gate = np.where(loss_diff < med_loss_diff)[0]
            # If no samples pass
            if gate.shape[0] == 0:
                message = 'Alpha: {}\nDelta_A: {}\nNo samples passed gradient gate!!!!\n \
                        Updating alpha and skipping...'.format(alpha, delta_a)
                print(message)
                with open(args.model_path + args.logfile, 'a') as f:
                    f.write(message + '\n')
                continue                
            
            gate = torch.LongTensor(gate) # Mask for the losses we will keep 
            gated = losses[gate.cuda()]
            loss = torch.sum(gated) / og_size
            loss_reported = loss.data[0] * og_size / gate.size(0)
            
            # Calculate ratio of samples passing loss
            ratio = float(gate.size(0)) / og_size
            print("Ratio of samples passing gradient: {}".format(ratio))


            print("Max loss: {}".format(torch.max(losses)))
            print("Max loss0: {}".format(torch.max(losses_0)))

            print("Min loss: {}".format(torch.min(losses)))
            print("Min loss0: {}".format(torch.min(losses_0)))

            print loss_reported
            batch_loss.append(loss_reported)

            loss.backward()
            optimizer.step()

            # Evaluate the model
            if i % args.val_step == 0:
                acc, gt_acc = test_func.bleu_test_acc(encoder, decoder, vocab)
                batch_acc.append((acc, gt_acc))


            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f, Ratio: %.4f, \
                        Val: %.5f, %.5f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss_reported, np.exp(loss_reported), ratio, acc, gt_acc)) 
                
                with open(args.model_path + args.logfile, 'a') as f:
                    f.write('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f, \
                            Ratio: %.2f, Val: %.5f, %.5f\n'
                          %(epoch, args.num_epochs, i, total_step, 
                            loss_reported, np.exp(loss_reported), ratio, acc, gt_acc)) 

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
                with open(args.model_path + 'training_val.pkl', 'wb+') as f:
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
    parser.add_argument('--val_step', type=int, default=100)

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
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--delta_a', type=float, default=1e-5)
    parser.add_argument('--log', action='store_true',
                        help='whether to gate based on log difference')
    parser.add_argument('--initial_a', type=float, default=1e-5,
                        help='specify desired initial value of alpha')
    parser.add_argument('--pass_margin', type=float, default=0.05,
                        help='specify far ratio can diverge from specified pass_ratio \
                                before alpha is updated')
    parser.add_argument('--train_encoder', action='store_true')
    parser.add_argument('--a_log_length', type=int, default=6)
    parser.add_argument('--divide' , action='store_true')
    parser.add_argument('--min_loss', type=float, default=1e-1)
    args = parser.parse_args()
    print(args)
    main(args)
