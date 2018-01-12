import torch
import argparse
import nltk
import numpy as np 
import pickle 
import os
from PIL import Image
from torch.autograd import Variable 
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from data_loader import get_loader
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

def decode(feature,user_input,decoder,vocab,c_step=0.0):
    sampled_ids = decoder.sample(feature,user_input,vocab,c_step=c_step)
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

def main(args):   
    with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    encoder = EncoderCNN(256)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(256, 512, 
                       len(vocab), 1)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder))
    decoder.load_state_dict(torch.load(args.decoder))

    bleu_score_origin, bleu_score_hint = bleu_test_acc(encoder, decoder, vocab, args.num_samples,
                                                       args.num_hints, args.debug, args.c_step)

    print "bleu score between output and ground true without hint\n"+str(bleu_score_origin)
    print "bleu score between output and ground true with hint\n"+str(bleu_score_hint)
    
def bleu_test_acc(encoder, decoder, vocab, num_samples=100, num_hints=2, debug=False, c_step=0.0):
    transform = transforms.Compose([
       transforms.ToTensor(), 
       transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    rt_image = './data/val_resized2014'
    annotations = './data/annotations/captions_val2014.json' 
    shuffle = False
    data_loader = get_loader(rt_image,
                  annotations,
                  vocab,
                  transform, 1, shuffle, 1)
    assert len(vocab) == decoder.linear.out_features
    bleu_score_origin=0
    bleu_score_hint=0
    #for i in range(0,len(data_loader)/1000):
    max_bleu = 0
    ref_sentence = 0
    hint = 0
    for i, (image, caption, length) in enumerate(data_loader):
        if i > num_samples:
            break
        image_tensor = to_var(image, volatile=True)
        caption = ' '.join([vocab.idx2word[c] for c in caption[0,1:-1]])
        feature = encoder(image_tensor)
        teach_wordid = [vocab.word2idx["<start>"]]
        for i in range(num_hints):
            teach_wordid.append(vocab.word2idx[caption.split()[i].lower()])
        # get the output with one word hint
        origin_sentence = decode(feature, teach_wordid[0:1], decoder, vocab, c_step=c_step)
        reference = caption.split()
        hypothesis = ' '.join(origin_sentence.split()[1:-1]) 
        no_hint = nltk.translate.bleu_score.sentence_bleu([caption], hypothesis)
        bleu_score_origin += no_hint

        hint_sentence = decode(feature,teach_wordid,decoder,vocab,c_step=c_step)
        hypothesis_hint = ' '.join(hint_sentence.split()[1:-1])
        hint = nltk.translate.bleu_score.sentence_bleu([caption], hypothesis_hint)
        bleu_score_hint += hint
        if debug:
            print("No hint: {}\nHint: {}\nBleu: {}\nBleu Improve: {}".format(origin_sentence, hint_sentence, no_hint, hint))
        if hint > max_bleu:
            
            max_bleu = hint
            max_sentence = hint_sentence
            ref_sentence = caption
    return bleu_score_origin/i, bleu_score_hint/i




        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str , default = './models/encoder-4-3000.pkl',
                        help='specify encoder')
    parser.add_argument('--decoder', type=str , default = './models/decoder-4-3000.pkl',
                        help='specify decoder')
    parser.add_argument('--test_set', action='store_true')
    parser.add_argument('--num_samples', type=int , default=500)
    parser.add_argument('--num_hints', type=int , default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--c_step', type=float , default=0.0)
    args = parser.parse_args()
    print(args)
    main(args)
