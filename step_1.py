import torch
import nltk
import matplotlib as plt
import numpy as np 
import argparse
import pickle 
import os
from torch.autograd import Variable 
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
from data_loader import CocoDataset
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

def decode(feature,user_input,decoder,vocab):
    sampled_ids = decoder.sample(feature,user_input)
    sampled_ids = sampled_ids.cpu().data.numpy()
    
    # Decode word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    return ' '.join(sampled_caption)


def main(args):   
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # data_loader = CocoDataset(root=args.image_dir,
    #                    json=args.caption_path,
    #                    vocab=vocab,
    #                    transform=transform)
    # i = 2
    # sampled_ids = data_loader[i][1]
    # sampled_caption = []
    # for word_id in sampled_ids:
    #     print word_id
    #     word = vocab.idx2word[word_id]
    #     sampled_caption.append(word)
    #     if word == '<end>':
    #         break
    # sentence = ' '.join(sampled_caption)
    # print (sentence)
    # print data_loader[i][0]
    # print data_loader[i][2]


    
    # Build Models
    encoder = EncoderCNN(args.embed_size)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare Image
    image = load_image(args.image, transform)
    image_tensor = to_var(image, volatile=True)
    
    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    
    # Generate caption from image
    feature = encoder(image_tensor)
    sentence = decode(feature,[],decoder,vocab)

    print (sentence)
    user_input = raw_input("Does it make sense to you?(y/n)\n")

    if str(user_input) == "n":
        f = open('data/step_1/caption_1.txt','r')
        ground_true = f.read()
        teach_wordid = []
        teach_wordid.append(vocab.word2idx["<start>"])
        while(True):
            print "This is the ground true:\n"+ground_true+"\n"+\
            "###################################################\n"
            reference = ground_true.split()
            hypothesis = sentence.split()
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
            print "Current BLEU score is "+str(BLEUscore)
            word = raw_input("next word:\n")
            word_idx = vocab.word2idx[word]
            teach_wordid.append(word_idx)
            sentence = decode(feature,teach_wordid,decoder,vocab)
            print "###################################################\n"
            print "Current Translated sentence is: \n"+sentence+"\n"


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default='./data/resized2014' ,
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='./models/encoder-4-3000.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-4-3000.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
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
    args = parser.parse_args()
    main(args)