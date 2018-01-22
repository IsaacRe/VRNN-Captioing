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

def prediction_diff(feature, user_input, decoder, vocab, c_step=0.0, debug=False):
    sample_ids, pred = decoder.sample(feature, user_input, vocab, c_step=c_step)
    
"""
Returns sampled word id's and prediction tensor of the last step
"""
def decode(feature,user_input,decoder,vocab,c_step=0.0):
    sampled_ids, predictions = decoder.sample(feature,user_input,vocab,c_step=c_step)
    sampled_ids = sampled_ids.cpu().data.numpy()
    
    # Decode word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    return ' '.join(sampled_caption), predictions

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

    prediction_diff = test(encoder, decoder, vocab, args.num_samples,
                                                       args.num_hints, args.debug, args.c_step)

    print "ground truth prediction difference without hint\n"+str(prediction_diff[0])
    print "ground truth prediction difference with hint\n"+str(prediction_diff[1])
    
def test(encoder, decoder, vocab, num_samples=100, num_hints=2, debug=False, c_step=0.0):
    transform = transforms.Compose([
       transforms.Resize(224),
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

    avg_gt_diff, avg_gt_diff_hint = 0, 0
    for i, (image, caption, length) in enumerate(data_loader):
        if i > num_samples:
            break
        image_tensor = to_var(image, volatile=True)
        caption = ' '.join([vocab.idx2word[c] for c in caption[0,1:-1]])
        feature = encoder(image_tensor)
        teach_wordid = [vocab.word2idx["<start>"]]
        for i in range(num_hints):
            if len(caption.split()) <= num_hints:
                break
            teach_wordid.append(vocab.word2idx[caption.split()[i].lower()])
        # get the output with no hint
        origin_sentence, pred_no_hint = decode(feature,[], decoder, vocab, c_step=c_step)
        # get the predictions for the step following last user input
        pred_no_hint = pred_no_hint[num_hints+args.skip_steps]

        hint_sentence, pred_hint = decode(feature,teach_wordid,decoder,vocab,c_step=c_step)
        # get the predictions for the step following last user input
        pred_hint = pred_hint[num_hints+args.skip_steps]
        
        # get the ground truth prediction tensor for the step following las user input
        gt_id = vocab.word2idx[caption.split()[num_hints+args.skip_steps-1]]

        # calculate difference between prediction scores for ground truth
        gt_diff = 1.0 - pred_no_hint[gt_id]
        gt_diff_hint = 1.0 - pred_hint[gt_id]


        avg_gt_diff += gt_diff
        avg_gt_diff_hint += gt_diff_hint

        if debug:
            print("Ground Truth: {}\nNo hint: {}\nHint: {}\nBleu: {}\nBleu Improve: {}\
                  \nGround Truth Score: {}\nGround Truth Score Improve {}\
                  ".format(caption, hypothesis, hypothesis_hint, no_hint, hint, 
                           gt_diff, gt_diff_hint))
    avg_gt_diff /= i
    avg_gt_diff_hint /= i
    return (avg_gt_diff, avg_gt_diff_hint)




        


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
    parser.add_argument('--skip_steps', type=int, default=1)
    args = parser.parse_args()
    print(args)
    main(args)
