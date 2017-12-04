import torch
import nltk
import numpy as np 
import pickle 
import os
from PIL import Image
from torch.autograd import Variable 
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
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

def encode(img,vocab):
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
    encoder = EncoderCNN(256)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    encoder.load_state_dict(torch.load('./models/encoder-4-3000.pkl'))
    image = load_image(img, transform)
    image_tensor = to_var(image, volatile=True)
    
    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda()
    feature = encoder(image_tensor)
    return feature

def main():   
    transform = transforms.Compose([
       transforms.ToTensor(), 
       transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    rt_image = './data/resized2014'
    data_loader = CocoDataset(root=rt_image,
                  json='./data/annotations/captions_train2014.json',
                  vocab=vocab,
                  transform=transform)
    encoder = EncoderCNN(256)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(256, 512, 
                       len(vocab), 1)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Load the trained model parameters
    encoder.load_state_dict(torch.load('./models/encoder-4-3000.pkl'))
    decoder.load_state_dict(torch.load('./models/decoder-4-3000.pkl'))
    bleu_score_origin=0
    bleu_score_hint=0
    for i in range(0,len(data_loader)/1000):
        image = load_image(rt_image+"/"+data_loader.retrieve(i)[0], transform)
        image_tensor = to_var(image, volatile=True)
        caption = data_loader.retrieve(i)[1]
        feature = encoder(image_tensor)
        teach_wordid = []
        teach_wordid.append(vocab.word2idx["<start>"])
        teach_wordid.append(vocab.word2idx[(caption.split()[0]).lower()])
        # get the output with one word hint
        origin_sentence = decode(feature,[],decoder,vocab)
        reference = caption.split()
        hypothesis = origin_sentence.split()
        bleu_score_origin += nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

        hint_sentence = decode(feature,teach_wordid,decoder,vocab)
        hypothesis = hint_sentence.split()
        bleu_score_hint += nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

    print "bleu score between output and ground true without hint\n"+str(bleu_score_origin/1000.0)
    print "bleu score between output and ground true with hint\n"+str(bleu_score_hint/1000.0)





        


if __name__ == '__main__':

    main()
