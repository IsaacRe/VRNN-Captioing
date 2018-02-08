import torch
import torch.nn as nn
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
from torch.nn.utils.rnn import pack_padded_sequence

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

def decode_beta(feature,user_input,decoder,vocab,c_step=0.0,prop_step=1):
    sampled_ids, predictions = decoder.sample_with_update(feature,user_input,vocab,c_step=c_step)
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

def crsEntropyLoss(caption,length, feature,vocab,num_hints,decoder,c_step,compare_steps):
    # Since sampled caption always has a length <= 20
    if length[0] > 20:
        length[0] = 20
    target = pack_padded_sequence(caption, length, batch_first=True)[0]
    target = to_var(target,volatile=True)
    # print "cap"
    # print caption
    caption = ' '.join([vocab.idx2word[c] for c in caption[0,1:-1]])
    # print "cap!"
    # print caption
    teach_wordid = [vocab.word2idx["<start>"]]
    for i in range(num_hints):
        if len(caption.split()) <= num_hints:
            break
        teach_wordid.append(vocab.word2idx[caption.split()[i].lower()])
    _, pred_no_hint = decoder.sample_beta(feature,[],vocab,c_step=c_step,prop_step=args.prop_steps)
    _, pred_hint = decoder.sample_beta(feature,teach_wordid,vocab,c_step=c_step,prop_step=args.prop_steps)
    pred_no_hint = to_var(pred_no_hint,volatile=True)
    pred_hint = to_var(pred_hint,volatile=True)
    criterion = nn.CrossEntropyLoss()
    # cut the predication matrix to have the same length as caption in order to compute loss
    return criterion(pred_no_hint[:length[0]], target), criterion(pred_hint[:length[0]],target)


def probabilityScore(caption,feature,vocab,num_hints,decoder,c_step,compare_steps):
    caption = ' '.join([vocab.idx2word[c] for c in caption[0,1:-1]])
    teach_wordid = [vocab.word2idx["<start>"]]
    for i in range(num_hints):
        if len(caption.split()) <= num_hints:
            break
        teach_wordid.append(vocab.word2idx[caption.split()[i].lower()])
    # get the output with no hint
    # origin_sentence, pred_no_hint = decode(feature,[], decoder, vocab, c_step=c_step)
    origin_sentence, pred_no_hint = decode_beta(feature,[], decoder, vocab, c_step=c_step, prop_step=args.prop_steps)

    # hint_sentence, pred_hint = decode(feature,teach_wordid,decoder,vocab,c_step=c_step)
    hint_sentence, pred_hint = decode_beta(feature,teach_wordid, decoder, vocab, c_step=c_step, prop_step=args.prop_steps)
    
    # get the ground truth ids for all steps following last user input
    gt_words = caption.split()[num_hints:]
    num_compare = min(len(gt_words), compare_steps)
    gt_ids = torch.LongTensor([vocab.word2idx[word] for word in gt_words[:num_compare]])
    
    # get the predictions for all steps following last user input
    pred_no_hint = pred_no_hint[num_hints:num_hints+num_compare]
    pred_hint = pred_hint[num_hints:num_hints+num_compare]

    # calculate prediction scores for ground truth
    gt_score = pred_no_hint.gather(1,gt_ids.view(-1,1))
    gt_score_hint = pred_hint.gather(1,gt_ids.view(-1,1))
    
    return gt_score,gt_score_hint,num_compare



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

    if args.test_prop0:
        decoder.test_h_from_c()
        return

    measurement_score = test(encoder, decoder, vocab, args.num_samples,
                                                       args.num_hints, args.debug, args.c_step)
    if args.msm == "ps":
        print "ground truth prediction score without hint\n"+str(measurement_score[0])
        print "ground truth prediction score with hint\n"+str(measurement_score[1])
    elif args.msm == "ce":
        print "Cross Entropy Loss without hint\n"+str(measurement_score[0])
        print "Cross Entropy Loss with hint\n"+str(measurement_score[1])

def test(encoder, decoder, vocab, num_samples, num_hints, debug=False, c_step=0.0):
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

    avg_gt_score, avg_gt_score_hint = torch.zeros(args.compare_steps,1), torch.zeros(args.compare_steps,1)

    avg_crossEnloss,avg_crossEnloss_hint = torch.cuda.FloatTensor(1),torch.cuda.FloatTensor(1)

    for i, (image, caption, length) in enumerate(data_loader):
        if i > num_samples:
            break
        image_tensor = to_var(image, volatile=True)
        feature = encoder(image_tensor)

        # Compute probability score
        if args.msm == "ps":
            gt_score, gt_score_hint,num_compare = probabilityScore(caption,feature,vocab,num_hints,decoder,c_step,args.compare_steps)
            avg_gt_score = avg_gt_score.index_add_(0, torch.LongTensor(range(num_compare)), gt_score)
            avg_gt_score_hint = avg_gt_score_hint.index_add_(0, torch.LongTensor(range(num_compare)), gt_score_hint)
        # Compute cross entropy loss
        elif args.msm == 'ce':
            crossEnloss, crossEnloss_hint = crsEntropyLoss(caption,length,feature,vocab,num_hints,decoder,c_step,args.compare_steps)
            avg_crossEnloss = avg_crossEnloss + crossEnloss.data
            avg_crossEnloss_hint = avg_crossEnloss_hint + crossEnloss_hint.data
        if debug:
            print("Ground Truth: {}\nNo hint: {}\nHint: {}\
                  \nGround Truth Score: {}\nGround Truth Score Improve {}\
                  ".format(caption, hypothesis, hypothesis_hint, gt_score, gt_score_hint))
    if args.msm == "ps":
        avg_gt_score /= i
        avg_gt_score_hint /= i
        return (avg_gt_score, avg_gt_score_hint)
    else:
        avg_crossEnloss /= i
        avg_crossEnloss_hint /= i
        return (avg_crossEnloss, avg_crossEnloss_hint)




        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str , default = './models/encoder_pretrained.pkl',
                        help='specify encoder')
    parser.add_argument('--decoder', type=str , default = './models/decoder_pretrained.pkl',
                        help='specify decoder')
    parser.add_argument('--test_set', action='store_true')
    parser.add_argument('--num_samples', type=int , default=2000)
    parser.add_argument('--num_hints', type=int , default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--c_step', type=float , default=0.0)
    parser.add_argument('--compare_steps', type=int , default=10)
    parser.add_argument('--prop_steps', type=int , default=1)
    parser.add_argument('--msm',type=str,default="ps",
        help='ps: probability score, ce: CrossEntropyLoss')
    parser.add_argument('--test_prop0', action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
