import torch
import torch.nn as nn
import argparse
import nltk
import numpy as np 
import pickle 
import os
from sys import path
import json
from PIL import Image
from torch.autograd import Variable 
from torchvision import transforms 
from utils.build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from utils.data_loader import get_loader
from collections import Counter
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torch.nn.utils.rnn import pack_padded_sequence

class CocoJson:

    def __init__(self, gt_json, pred_json):
        self.gt_json = gt_json
        self.pred_json = pred_json
        self.idncaption = []
        self.idnimage = []
        self.idnprediction = []
        self.imgids = []

    def add_entry(self, img_id=1, ann_id=1, caption=None, pred_caption=''):
        if img_id in self.imgids:
            return
        self.imgids.append(img_id)
        
        if caption is not None:
            temp = { \
                    "image_id": img_id,
                    "id": ann_id,
                    "caption": caption,
                    }

            temp2 = { "id": img_id }

        temp3 = { \
                "image_id": img_id,
                "id": ann_id,
                "caption": pred_caption,
                }

        if caption is not None:
            self.idncaption.append(temp)
            self.idnimage.append(temp2)

        self.idnprediction.append(temp3)

    def new_pred(self, img_id=1, ann_id=1, pred_caption=''):
        assert img_id in imgids
        self.idnprediction = [{ \
                              "image_id": img_id,
                              "id": ann_id,
                              "caption": pred_caption,
                              }]
        self.create_json()

    
    def create_json(self):
        if self.idncaption != []:
            formatted = { \
                        "type": "", "info": "", "licenses": None,
                        "annotations": self.idncaption,
                        "images": self.idnimage,
                        }
            with open(self.gt_json, 'w+') as f:
                json.dump(formatted, f)

        with open(self.pred_json, 'w+') as f:
            json.dump(self.idnprediction, f)

        self.idncaption = []
        self.idnimage = []
        self.idnprediction = []

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

"""
Returns sampled word id's and prediction tensor of the last step
"""

def decode_beta(feature,user_input,decoder,vocab,c_step=0.0,prop_step=1):
    sampled_ids, predictions = decoder.sample(feature,user_input,vocab,c_step=c_step, prop_step=prop_step, update_method=args.update_method)
    sampled_ids = sampled_ids.numpy()
    
    # Decode word_ids to words
    sampled_caption = []
    sampled_caption_no_update = []
    for word_id in sampled_ids[0]:
        word = vocab.idx2word[word_id]
        sampled_caption_no_update.append(word)
        if word == '<end>':
            break
    for word_id in sampled_ids[1]:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break

    return ' '.join(sampled_caption_no_update[1:-1]), ' '.join(sampled_caption[1:-1]), predictions

def semantic_similarity(decoder,vocab,str1,str2):
    assert len(str1)==len(str2), "two sentences should have the same length"
    similarity = 0
    pdist = nn.CosineSimilarity()
    for word1,word2 in zip(str1,str2):
        embed1 = decoder.embed(to_var(torch.LongTensor([vocab.word2idx[word1]])))
        embed2 = decoder.embed(to_var(torch.LongTensor([vocab.word2idx[word2]])))
        similarity += pdist(embed1, embed2)
    return similarity/len(str1)


def cocoEval(val='../data/captions_val2014.json', res='../data/captions_val2014_results.json'):
    coco = COCO(val)
    cocoRes = coco.loadRes(res)

    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    cocoEval.evaluate()

    scores = {}
    for metric, score in cocoEval.eval.items():
        scores[metric] = score

    if __name__ == '__main__':
        with open(args.filepath, 'w+') as f:
            pickle.dump(scores, f)

    return scores


def main(args):
    with open('../data/vocab.pkl', 'rb') as f:
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


    measurement_score = test(encoder, decoder, vocab, args.num_samples,
                                            args.num_hints, args.debug, args.c_step, args.no_avg)
    if args.msm == "co":
        scores = cocoEval()
        scores_u = cocoEval(res='../data/captions_val2014_results_u.json')
        print(scores)
        print(scores_u)

def test(encoder, decoder, vocab, num_samples, num_hints, debug=False, c_step=0.0, no_avg=True):
    transform = transforms.Compose([
       transforms.Resize(224),
       transforms.ToTensor(), 
       transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    rt_image = '../data/val_resized2014'
    annotations = args.caption or '../data/annotations/captions_val2014.json' 
    shuffle = False
    data_loader = get_loader(rt_image,
                  annotations,
                  vocab,
                  transform, 1, shuffle, 1)
    assert len(vocab) == decoder.linear.out_features

    avg_gt_score, avg_gt_score_hint = torch.zeros(args.compare_steps,1), torch.zeros(args.compare_steps,1)
    gt_scores, gt_scores_hint = [], []

    avg_crossEnloss,avg_crossEnloss_hint = torch.zeros(args.compare_steps,1), torch.zeros(args.compare_steps,1)
    crossEnlosses, crossEnlosses_hint = [], []

    num_sampled = 0
    data_points = []
    coco_json = CocoJson('../data/captions_val2014.json', '../data/captions_val2014_results.json')
    coco_json_update = CocoJson('../data/captions_val2014.json', '../data/captions_val2014_results_u.json')

    count = 0
    for i, (image, caption, length, img_id, ann_id) in enumerate(data_loader):
        if num_sampled > num_samples or i > num_samples:
            break
        image_tensor = to_var(image, volatile=True)
        feature = encoder(image_tensor)

        # Compute optimal c_step by (pred, ce)
        if args.msm == "co":
            no_update, pred_caption, _ = decode_beta(feature, caption[0,:num_hints+1], decoder, \
                                          vocab, c_step, args.prop_steps)
            # print caption
            # no_hint, _, _ = decode_beta(feature,caption[0,:1], decoder, \
            #                                           vocab, c_step, args.prop_steps)

            caption = [vocab.idx2word[c] for c in caption[0,1:-1]]

            no_update = ' '.join(caption[:num_hints]) + ' ' + ' '.join(no_update.split()[num_hints:])
            pred_caption = ' '.join(caption[:num_hints]) + ' ' + ' '.join(pred_caption.split()[num_hints:])
            caption = ' '.join(caption)

            if args.load_val:
                caption = None

            coco_json_update.add_entry(img_id[0], ann_id[0], caption, pred_caption)
            coco_json.add_entry(img_id[0], ann_id[0], caption, no_update)

        if debug:
            print("Ground Truth: {}\nNo hint: {}\nHint: {}\
                  \nGround Truth Score: {}\nGround Truth Score Improve {}\
                  ".format(caption, hypothesis, hypothesis_hint, gt_score, gt_score_hint))

    if args.msm == "co":
        coco_json.create_json()
        coco_json_update.create_json()
        return None

        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str , default = '../models/encoder_pretrained.pkl',
                        help='specify encoder')
    parser.add_argument('--decoder', type=str , default = '../models/decoder_pretrained.pkl',
                        help='specify decoder')
    parser.add_argument('--test_set', action='store_true')
    parser.add_argument('--caption', type=str , default = '../data/annotations/captions_val2014.json')
    parser.add_argument('--num_samples', type=int , default=4000)
    parser.add_argument('--num_hints', type=int , default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--c_step', type=float , default=2.0)
    parser.add_argument('--compare_steps', type=int , default=10)
    parser.add_argument('--prop_steps', type=int , default=-1)
    parser.add_argument('--msm',type=str,default="co",
            help='ps: probability score, ce: CrossEntropyLoss, co: cocoEval')
    parser.add_argument('--no_avg', action='store_true')
    parser.add_argument('--filepath', type=str , default='../hint_improvement.pkl')
    parser.add_argument('--update_step', type=int , default=0)
    parser.add_argument('--update_method', type=str , default='c')
    parser.add_argument('--load_val', action='store_true',
            help='if true, loads gt for all samples into json file in data dir')
    args = parser.parse_args()
    print(args)
    main(args)
