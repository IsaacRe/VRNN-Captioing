import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
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

def log_string(log_path, out_str):
    log_path.write(out_str+'\n')
    log_path.flush()
    print(out_str)
    
def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

"""
Returns sampled word id's, currently not in use.
"""
def decode(sample_ids, vocab):
    sampled_ids = sampled_ids.cpu().data.numpy()
    
    # Decode word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    return ' '.join(sampled_caption)

def train_one_epoch(data_loader_train, encoder, decoder, criterion, optimizer, epoch, args, log_fout):
    # Train the Models
    total_step_train = len(data_loader_train)
    
    num_iterations = args.num_train_samples/args.batch_size
    losses = []
    for i, (images, captions, lengths, _, _) in enumerate(data_loader_train):
        if num_iterations < i:
            log_string(log_fout, "Trained %d samples"%(num_iterations * args.batch_size))
            break
        # Set mini-batch dataset
        # images are in shape of 128x3x224x224
        images = to_var(images, volatile=True)
        # captions are in shape of 128xLongestSeq
        captions = to_var(captions)
        # padded with the longest sequence at lengths[0], 128xLongestSeq
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Now, we use the current weight to predict new captions as input and do a back propagation
        # We want it to train on the entire sequence even if there are hints.
        # Forward, Backward and Optimize
        decoder.zero_grad()
        encoder.zero_grad()
        # features are in shape of 128x256
        features = encoder(images)

        # ==========Original forward function==================
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        """ Our forward method
        # Print log info
        if i % args.log_step == 0:            
            print ('Done training the first step, now training with introspective forwarding for epoch %d'%epoch)
        
        # Forward, Backward and Optimize
        decoder.zero_grad()
        encoder.zero_grad()
        # features are in shape of 128x256
        features = encoder(images)

        # ==========Our forward function=======================
        # forward with all hints.
        # The same function should be using instrospective_forward but with enough hints.
        outputs = decoder.introspective_forward(features, captions, lengths, hints = args.num_hints)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        """
        losses.append(loss.data[0])
        
        # Print log info
        if i % args.log_step == 0:
            log_string(log_fout, 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                  %(epoch, args.num_epochs, i, min(num_iterations, total_step_train), 
                    loss.data[0], np.exp(loss.data[0]))) 

        # Save the models
        if (i+1) % args.save_step == 0:
            torch.save(decoder.state_dict(), 
                       os.path.join(args.model_path, 
                                    'decoder-%d-%d.pkl' %(epoch+1, i+1)))
            torch.save(encoder.state_dict(), 
                       os.path.join(args.model_path, 
                                    'encoder-%d-%d.pkl' %(epoch+1, i+1)))

    log_string(log_fout, 'Avg Train Loss: %.4f'%(np.mean(losses)))
    torch.save(decoder.state_dict(), 
               os.path.join(args.model_path, 
                            'decoder-%d-epoch.pkl' %(epoch+1)))
    torch.save(encoder.state_dict(), 
               os.path.join(args.model_path, 
                            'encoder-%d-epoch.pkl' %(epoch+1)))
    
def eval_one_epoch(data_loader_eval, encoder, decoder, criterion, epoch, args, log_fout):
    losses = []
    total_step_eval = len(data_loader_eval)
    num_iterations = args.num_eval_samples/args.batch_size
    for i, (images, captions, lengths, _, _) in enumerate(data_loader_eval):
        if num_iterations < i:
            log_string(log_fout, "Evaluated %d samples"%(num_iterations * args.batch_size))
            break
        # captions are in shape of 1xLongestSeq
        captions = to_var(captions)
        # padded with the longest sequence at lengths[0], 1xLongestSeq
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        image_tensor = to_var(images, volatile=True)
        features = encoder(image_tensor)
        outputs = decoder.introspective_forward(features, captions, lengths, hints=args.num_hints)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
    log_string(log_fout,  'Avg Eval Loss: %.4f'%(np.mean(losses)))

def main(args, log_fout):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform_train = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # For evaluation
    transform_val = transforms.Compose([
       transforms.Resize(224),
       transforms.ToTensor(), 
       transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader_train = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform_train, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    
    # Add evaluation loader
    data_loader_eval = get_loader('./data/val_resized2014',
                                  './data/annotations/captions_val2014.json',
                                  vocab, transform_val, args.batch_size, False, args.num_workers)
    
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
    
    
    log_string(log_fout, "We are using the number of hints = %d"%args.num_hints)
    log_string(log_fout, "[Start Time] " + str(datetime.now()))
    for epoch in range(args.num_epochs):
        log_string(log_fout, "====================================================================")
        t0 = datetime.now()
        train_one_epoch(data_loader_train, encoder, decoder, criterion, optimizer, epoch, args, log_fout)
        t1 = datetime.now()
        log_string(log_fout, "Train time Usage: " + str(t1 - t0))
        eval_one_epoch(data_loader_eval, encoder, decoder, criterion, epoch, args, log_fout)
        t2 = datetime.now()
        log_string(log_fout, "Eval time Usage: " + str(t2 - t1))

    log_string(log_fout,  "[End Time] "+ str(datetime.now()))
    
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
    parser.add_argument('--num_hints', type=int, default=-1)
    parser.add_argument('--log_path', type=str, default='./logs/train_log.txt')
    parser.add_argument('--num_train_samples', type=int, default=1024) # There is roughly a total of 414114 training samples
    parser.add_argument('--num_eval_samples', type=int, default=256) # There is roughly a total of 202654 testing samples
    
    args = parser.parse_args()
    
    dirs = os.path.dirname(args.log_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        
    log_fout = open(args.log_path, 'w')
    log_string(log_fout, str(args))
    main(args, log_fout)
