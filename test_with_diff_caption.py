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
    sampled_ids, predictions = decoder.sample(feature,user_input,vocab,c_step=c_step, prop_step=prop_step, update_method=args.update_method)
    sampled_ids = sampled_ids.numpy()
    
    # Decode word_ids to words
    sampled_caption = []
    sampled_caption_no_update = []
    for word_id in sampled_ids:
        isEnd = isEndu = False
        word = '<end>' if isEnd else vocab.idx2word[word_id[0]]
        sampled_caption_no_update.append(word)
        wordu = '<end>' if isEndu else vocab.idx2word[word_id[1]]
        sampled_caption.append(wordu)
        if word == '<end>': isEnd = True
        if wordu == '<end>': isEndu = True
        if isEnd and isEndu:
            break

    return ' '.join(sampled_caption_no_update[1:-1]), ' '.join(sampled_caption[1:-1]), predictions


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

def crsEntropyLoss(caption,length, feature,vocab,num_hints,decoder,c_step,compare_steps):
    # Since sampled caption always has a length <= 20
    if num_hints + compare_steps + 1 > min(length[0], 20):
        if num_hints + 1 < min(length[0], 20):
            compare_steps = min(length[0], 20) - 1 - num_hints
        else:
            print(' '.join([vocab.idx2word[i] for i in caption]))
            return None, None
    target = pack_padded_sequence(caption[:,1+num_hints:1+num_hints+compare_steps], length, batch_first=True)[0]
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
    _, _, predictions = decoder.sample(feature,teach_wordid,vocab,c_step=c_step,prop_step=args.prop_steps,update_method=args.update_method)
    pred_no_update = to_var(predictions[0],volatile=True)
    pred_update = to_var(predictions[1],volatile=True)
    criterion = nn.CrossEntropyLoss()
    result, result_update = [], []
    for i in range(len(target)):
        result.append(criterion(pred_no_update[num_hints+1+i:num_hints+2+i], target[i:i+1]).cpu().data)
        result_update.append(criterion(pred_update[num_hints+1+i:num_hints+2+i], target[i:i+1]).cpu().data)
    result, result_update = torch.cat(result, 0), torch.cat(result_update, 0)
    # cut the predication matrix to have the same length as caption in order to compute loss
    assert result.size(0) == compare_steps
    return result.unsqueeze(1), result_update.unsqueeze(1), compare_steps 


def probabilityScore(caption,feature,vocab,num_hints,decoder,c_step,compare_steps):
    caption = ' '.join([vocab.idx2word[c] for c in caption[0,1:-1]])
    teach_wordid = [vocab.word2idx["<start>"]]
    for i in range(num_hints):
        if len(caption.split()) <= num_hints:
            break
        teach_wordid.append(vocab.word2idx[caption.split()[i].lower()])

    # hint_sentence, pred_hint = decode(feature,teach_wordid,decoder,vocab,c_step=c_step)
    _, _, predictions = decode_beta(feature,teach_wordid, decoder, vocab, c_step=c_step, prop_step=args.prop_steps)
    
    # get the ground truth ids for all steps following last user input
    gt_words = caption.split()[num_hints:]
    num_compare = min(len(gt_words), compare_steps)
    gt_ids = torch.LongTensor([vocab.word2idx[word] for word in gt_words[:num_compare]])
    
    # get the predictions for all steps following last user input
    pred_no_hint = predictions[0,num_hints+1:num_hints+1+num_compare] # <start> provided in user_input
    pred_hint = predictions[1,num_hints+1:num_hints+1+num_compare]

    # calculate prediction scores for ground truth
    gt_score = pred_no_hint.gather(1,gt_ids.view(-1,1))
    gt_score_hint = pred_hint.gather(1,gt_ids.view(-1,1))
    
    return gt_score, gt_score_hint, num_compare


def cocoEval(val='data/captions_val2014.json', res='data/captions_val2014_results.json'):
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

    if args.test_c_step:
        data_points = test(encoder, decoder, vocab, args.num_samples, args.num_hints)

        with open(args.filepath, 'w+') as f:
            pickle.dump(data_points, f)

        print("Done sampling for c_step evaluation. Data saved to {}".format(args.filepath))

        return

    measurement_score = test(encoder, decoder, vocab, args.num_samples,
                                            args.num_hints, args.debug, args.c_step, args.no_avg)
    if args.msm == "ps":
        if not args.no_avg:
            print "ground truth prediction score without update\n"+str(measurement_score[0])
            print "ground truth prediction score with update\n"+str(measurement_score[1])
            print "Difference\n"+str(measurement_score[1]-measurement_score[0])
        else:
            with open(args.filepath, 'w+') as f:
                pickle.dump(measurement_score, f)
            print "Done. Data saved to {}".format(args.filepath)
    elif args.msm == "ce":
        if not args.no_avg:
            print "Cross Entropy Loss without update\n"+str(measurement_score[0])
            print "Cross Entropy Loss with update\n"+str(measurement_score[1])
            print "Difference\n"+str(measurement_score[1]-measurement_score[0])
        else:
            with open(args.filepath, 'w+') as f:
                pickle.dump(measurement_score, f)
            print "Done. Data saved to {}".format(args.filepath)
    elif args.msm == "co":
        scores = cocoEval()
        scores_u = cocoEval(res='data/captions_val2014_results_u.json')
        print(scores)
        print(scores_u)

def test(encoder, decoder, vocab, num_samples, num_hints, debug=False, c_step=0.0, no_avg=True):
    transform = transforms.Compose([
       transforms.Resize(224),
       transforms.ToTensor(), 
       transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    rt_image = './data/val_resized2014'
    annotations = args.caption or './data/annotations/captions_val2014.json' 
    shuffle = False
    batch_size = 2 if args.adapt else 1 # If inputting random caption as gt, use batch of 2 and swap gts
    data_loader = get_loader(rt_image,
                  annotations,
                  vocab,
                  transform, batch_size, shuffle, 1)
    assert len(vocab) == decoder.linear.out_features

    avg_gt_score, avg_gt_score_hint = torch.zeros(args.compare_steps,1), torch.zeros(args.compare_steps,1)
    gt_scores, gt_scores_hint = [], []

    avg_crossEnloss,avg_crossEnloss_hint = torch.zeros(args.compare_steps,1), torch.zeros(args.compare_steps,1)
    crossEnlosses, crossEnlosses_hint = [], []

    num_sampled = 0
    data_points = []
    coco_json = CocoJson('data/captions_val2014.json', 'data/captions_val2014_results.json')
    coco_json_update = CocoJson('data/captions_val2014.json', 'data/captions_val2014_results_u.json')

    for i, (images, captions, lengths, img_ids, ann_ids) in enumerate(data_loader):
        if i >= num_samples or args.adapt and i*2 >= num_samples:
            break

        for k in range(batch_size):
            image, length, img_id, ann_id = images[k:k+1], lengths[k:k+1], \
                                            img_ids[k:k+1], ann_ids[k:k+1]

            caption = captions[k:k+1]
            if args.adapt: # use the other image's caption for gt input
                gt_input = captions[(k+1) % batch_size, : args.num_hints + 1]
            else:
                gt_input = captions[k, : args.num_hints + 1]

            image_tensor = to_var(image, volatile=True)
            feature = encoder(image_tensor)

            # Compute probability score
            if args.msm == "ps":
                gt_score, gt_score_hint, num_compare = probabilityScore(caption,feature,vocab,num_hints,decoder,c_step,args.compare_steps)
                if not no_avg:
                    avg_gt_score = avg_gt_score.index_add_(0, torch.LongTensor(range(num_compare)), gt_score)
                    avg_gt_score_hint = avg_gt_score_hint.index_add_(0, torch.LongTensor(range(num_compare)), gt_score_hint)
                else:
                    gt_scores.append(gt_score[:num_compare])
                    gt_scores_hint.append(gt_score_hint[:num_compare])
            # Compute cross entropy loss
            elif args.msm == 'ce':
                crossEnloss, crossEnloss_hint, num_compare = crsEntropyLoss(caption,length,feature,vocab,num_hints,decoder,c_step,args.compare_steps)
                if type(crossEnloss) == type(None):
                    continue
                if not no_avg:
                    avg_crossEnloss = avg_crossEnloss.index_add_(0, torch.LongTensor(range(num_compare)), crossEnloss)
                    avg_crossEnloss_hint = avg_crossEnloss_hint.index_add_(0, torch.LongTensor(range(num_compare)), crossEnloss_hint)
                else:
                    crossEnlosses.append(crossEnloss)
                    crossEnlosses_hint.append(crossEnloss_hint)
            # Evaluate with pycoco tools
            elif args.msm == "co":
                no_update, pred_caption, _ = decode_beta(feature, gt_input, decoder, \
                                              vocab, c_step, args.prop_steps)
                
                caption = [vocab.idx2word[c] for c in caption[0,1:-1]]
                gt_input = [vocab.idx2word[c] for c in gt_input[:-1]]
                no_update = ' '.join(gt_input) + ' ' + ' '.join(no_update.split()[num_hints:])
                pred_caption = ' '.join(gt_input) + ' ' + ' '.join(pred_caption.split()[num_hints:])
                caption = ' '.join(caption)

                if args.load_val:
                    caption = None

                coco_json_update.add_entry(img_id[0], ann_id[0], caption, pred_caption)
                coco_json.add_entry(img_id[0], ann_id[0], caption, no_update)

            if debug and not args.test_c_step:
                print("Ground Truth: {}\nNo hint: {}\nHint: {}\
                      \nGround Truth Score: {}\nGround Truth Score Improve {}\
                      ".format(caption, hypothesis, hypothesis_hint, gt_score, gt_score_hint))

    if args.test_c_step:
        return data_points

    if args.msm == "ps":
        avg_gt_score /= i
        avg_gt_score_hint /= i
        if not no_avg:
            return (avg_gt_score, avg_gt_score_hint)
        else:
            return (gt_scores, gt_scores_hint)
    elif args.msm == "ce":
        avg_crossEnloss /= i
        avg_crossEnloss_hint /= i
        if not no_avg:
            return (avg_crossEnloss, avg_crossEnloss_hint)
        else:
            return (crossEnlosses, crossEnlosses_hint)
    elif args.msm == "co":
        coco_json.create_json()
        coco_json_update.create_json()
        return None

        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str , default = './models/encoder_pretrained.pkl',
                        help='specify encoder')
    parser.add_argument('--decoder', type=str , default = './models/decoder_pretrained.pkl',
                        help='specify decoder')
    parser.add_argument('--test_set', action='store_true')
    parser.add_argument('--caption', type=str , default = './data/annotations/captions_val2014.json')
    parser.add_argument('--num_samples', type=int , default=2000)
    parser.add_argument('--num_hints', type=int , default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--c_step', type=float , default=0.0)
    parser.add_argument('--compare_steps', type=int , default=10)
    parser.add_argument('--prop_steps', type=int , default=-1)
    parser.add_argument('--msm',type=str,default="co",
            help='ps: probability score, ce: CrossEntropyLoss, co: cocoEval')
    parser.add_argument('--test_prop0', action='store_true')
    parser.add_argument('--test_c_step', action='store_true')
    parser.add_argument('--no_avg', action='store_true')
    parser.add_argument('--filepath', type=str , default='hint_improvement.pkl')
    parser.add_argument('--update_step', type=int , default=0)
    parser.add_argument('--update_method', type=str , default='c')
    parser.add_argument('--load_val', action='store_true',
            help='if true, loads gt for all samples into json file in data dir')
    parser.add_argument('--adapt', action='store_true',
            help='adapt prediction mode, gt is sampled randomly from other images')
    args = parser.parse_args()
    print(args)
    main(args)
