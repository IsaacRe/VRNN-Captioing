import torch
import nltk
import matplotlib as plt
import numpy as np 
import argparse
import pickle 
import os
import sys
from torch.autograd import Variable 
from torchvision import transforms 
from utils.build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
from utils.data_loader import CocoDataset
from collections import Counter
from pycocotools.coco import COCO

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def decode(feature,user_input,decoder,vocab,c_step=0.0,prop_step=1):
    sampled, _ = decoder.sample(feature,torch.cuda.LongTensor(user_input),vocab,c_step=c_step,prop_step=prop_step)
    sampled_ids, sampled_ids_u = sampled[0].numpy(), sampled[1].numpy()
    
    # Decode word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    # Decode word_ids_u to words
    sampled_caption_u = []
    for word_id in sampled_ids_u:
        word = vocab.idx2word[word_id]
        sampled_caption_u.append(word)
        if word == '<end>':
            break

    return ' '.join(sampled_caption), ' '.join(sampled_caption_u)

def decode_word(feature,user_input,decoder,vocab):
    sampled_ids = decoder.next_word(feature,user_input,3)
    sampled_ids = sampled_ids.cpu().data.numpy()
    
    # Decode word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    return sampled_caption

def encode(img,vocab,):
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
    encoder = EncoderCNN(256)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    encoder.load_state_dict(torch.load('../models/encoder_pretrained.pkl'))
    image = load_image(img, transform)
    image_tensor = to_var(image, volatile=True)
    
    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda()
    feature = encoder(image_tensor)
    return feature


def main(args):   
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build Models
    encoder = EncoderCNN(args.embed_size)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare Image
    image = load_image('../application/static/candidate/image_candidates/'+args.image+'.jpg', transform)
    image_tensor = to_var(image, volatile=True)
    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    
    # Generate caption from image
    feature = encoder(image_tensor)
    sentence = decode(feature,[vocab.word2idx["<start>"]],decoder,vocab, c_step=args.c_step,prop_step=args.prop_step)

    print (sentence[1])
    user_input = raw_input("Does it make sense to you?(y/n)\n")

    if str(user_input) == "n":
        f = open('../application/static/candidate/caption_candidates/'+args.image+'.txt','r')
        ground_true = f.read().lower()
        teach_wordid = []
        teach_word = []
        teach_wordid.append(vocab.word2idx["<start>"])
        teach_word.append("<start>")
        while(True):
            print "This is the ground true:\n"+ground_true+"\n"+\
            "###################################################\n"
            
            word = raw_input("next word:\n")
            if word.lower() not in vocab.word2idx:
                print "Word is not in the vocabulary, please try another one!"
                continue
            teach_word.append(word.lower())
            word_idx = vocab.word2idx[word.lower()]
            teach_wordid.append(word_idx)
            sentence = decode(feature,teach_wordid,decoder,vocab, c_step=args.c_step,prop_step=args.prop_step)
            print "###################################################\n"
            complete_sentence =  ' '.join(teach_word+sentence[1].split()[len(teach_wordid):])
            print "Current Translated sentence is: \n"+complete_sentence+"\n"
    



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default='../data/resizedVal2014' ,
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_val2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='../models/encoder_pretrained.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='../models/decoder_pretrained.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab.pkl',
                        help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--user_input',type=bool,default=True,
                        help='user input starting words')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--c_step', type=float, default=0.0)
    parser.add_argument('--prop_step', type=int, default=1)
    args = parser.parse_args()
    main(args)
