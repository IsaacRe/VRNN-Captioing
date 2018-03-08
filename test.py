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
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from data_loader import get_loader
from collections import Counter
from pycocotoolscap.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
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
    sampled_ids, predictions = decoder.sample_beta(feature,user_input,vocab,c_step=c_step, prop_step=prop_step, update_method=args.update_method)
    sampled_ids = sampled_ids.numpy()
    
    # Decode word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    return ' '.join(sampled_caption[1:-1]), predictions


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
    _, pred_no_hint = decoder.sample_beta(feature,teach_wordid,vocab,c_step=0.0,prop_step=args.prop_steps,update_method=args.update_method)
    _, pred_hint = decoder.sample_beta(feature,teach_wordid,vocab,c_step=c_step,prop_step=args.prop_steps,update_method=args.update_method)
    pred_no_hint = to_var(pred_no_hint,volatile=True)
    pred_hint = to_var(pred_hint,volatile=True)
    criterion = nn.CrossEntropyLoss()
    result, result_hint = [], []
    for i in range(len(target)):
        result.append(criterion(pred_no_hint[num_hints+1+i:num_hints+2+i], target[i:i+1]).cpu().data)
        result_hint.append(criterion(pred_hint[num_hints+1+i:num_hints+2+i], target[i:i+1]).cpu().data)
    result, result_hint = torch.cat(result, 0), torch.cat(result_hint, 0)
    # cut the predication matrix to have the same length as caption in order to compute loss
    assert result.size(0) == compare_steps
    return result.unsqueeze(1), result_hint.unsqueeze(1), compare_steps 


def probabilityScore(caption,feature,vocab,num_hints,decoder,c_step,compare_steps):
    caption = ' '.join([vocab.idx2word[c] for c in caption[0,1:-1]])
    teach_wordid = [vocab.word2idx["<start>"]]
    for i in range(num_hints):
        if len(caption.split()) <= num_hints:
            break
        teach_wordid.append(vocab.word2idx[caption.split()[i].lower()])
    # get the output with no hint
    # origin_sentence, pred_no_hint = decode(feature,[], decoder, vocab, c_step=c_step)
    origin_sentence, pred_no_hint = decode_beta(feature,teach_wordid, decoder, vocab, c_step=0.0, prop_step=args.prop_steps)

    # hint_sentence, pred_hint = decode(feature,teach_wordid,decoder,vocab,c_step=c_step)
    hint_sentence, pred_hint = decode_beta(feature,teach_wordid, decoder, vocab, c_step=c_step, prop_step=args.prop_steps)
    
    # get the ground truth ids for all steps following last user input
    gt_words = caption.split()[num_hints:]
    num_compare = min(len(gt_words), compare_steps)
    gt_ids = torch.LongTensor([vocab.word2idx[word] for word in gt_words[:num_compare]])
    
    # get the predictions for all steps following last user input
    pred_no_hint = pred_no_hint[num_hints+1:num_hints+1+num_compare] # <start> provided in user_input
    pred_hint = pred_hint[num_hints+1:num_hints+1+num_compare]

    # calculate prediction scores for ground truth
    gt_score = pred_no_hint.gather(1,gt_ids.view(-1,1))
    gt_score_hint = pred_hint.gather(1,gt_ids.view(-1,1))
    
    return gt_score, gt_score_hint, num_compare

def createJson(predicted, ground_truth=None):
    with open('data/captions_val2014_results.json', 'w+') as f:
        json.dump(predicted, f)
    if ground_truth is not None:
        with open('data/captions_val2014.json', 'w+') as f:
            json.dump(ground_truth, f)

def cocoEval():
    coco = COCO('data/captions_val2014.json')
    cocoRes = coco.loadRes('data/captions_val2014_results.json')

    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    cocoEval.evaluate()

    scores = {}
    for metric, score in cocoEval.eval.items():
        scores[metric] = score

    with open(args.filepath, 'w+') as f:
        pickle.dump(scores, f)




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
        cocoEval()

def test(encoder, decoder, vocab, num_samples, num_hints, debug=False, c_step=0.0, no_avg=True):
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
    gt_scores, gt_scores_hint = [], []

    avg_crossEnloss,avg_crossEnloss_hint = torch.zeros(args.compare_steps,1), torch.zeros(args.compare_steps,1)
    crossEnlosses, crossEnlosses_hint = [], []

    num_sampled = 0
    data_points = []
    idnimage = []
    idncaption = []
    idnprediction = []
    imgids = []

    for i, (image, caption, length, img_id, ann_id) in enumerate(data_loader):
        if num_sampled > num_samples or i > num_samples and not args.test_c_step:
            break

        image_tensor = to_var(image, volatile=True)
        feature = encoder(image_tensor)

        # Compute optimal c_step by (pred, ce)
        if args.test_c_step:
            c_steps = list(np.exp(np.arange(0.1, 4, 0.05))-1)
            user_input = caption[0,1:-1]
            update_step = np.random.randint(2,7) if args.update_step == 0 else args.update_step

            p_score, ce_score = decoder.sample_with_update(feature, user_input, vocab, None, c_steps, args.compare_steps, args.update_method, update_step)

            # determine optimal c_step, dependent on p_score/ce_score of predictions at update step
            if type(p_score) == type(None):
                continue

            assert p_score.size(0) == len(c_steps) + 1 and p_score.size() == ce_score.size()
            
            if p_score.size(1) == args.compare_steps + 1 and ce_score.size(1) == p_score.size(1):
                num_sampled += 1
            

            # return [optimal c_steps wrt p_score], p_score of updated step, 
            #        [optimal c_steps wrt ce_score], ce_score of updated step
            # (2 * p_score.size(1) data points)

            data_points.append(( [([0.0]+c_steps)[j] for j in p_score.max(0)[1]], p_score[0,0], \
                                 [([0.0]+c_steps)[j] for j in ce_score.max(0)[1]], ce_score[0,0] ))

        # Compute probability score
        elif args.msm == "ps":
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
            no_update, _ = decode_beta(feature, caption[0,:num_hints+1], decoder, \
                                       vocab, 0.0, args.prop_steps)
            
            pred_caption, _ = decode_beta(feature, caption[0,:num_hints+1], decoder, \
                                          vocab, c_step, args.prop_steps)
            
            caption = [vocab.idx2word[c] for c in caption[0,1:-1]]
            no_update = ' '.join(caption[:num_hints]) + ' ' + ' '.join(no_update.split()[num_hints:])
            pred_caption = ' '.join(caption[:num_hints]) + ' ' + ' '.join(pred_caption.split()[num_hints:])
            caption = ' '.join(caption)

            if args.load_val:
                temp1 = dict()
                temp1["image_id"] = img_id[0]
                temp1["id"] = ann_id[0]
                temp1["caption"] = caption

                temp2 = dict()
                temp2["id"] = img_id[0]
            
            temp3 = dict()
            temp3["image_id"] = img_id[0]
            temp3["id"] = ann_id[0]
            temp3["caption"] = pred_caption

            if img_id[0] not in imgids:
                if args.load_val:
                    idncaption.append(temp1)
                    idnimage.append(temp2)
                idnprediction.append(temp3)
                imgids.append(img_id[0])

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
        val_data = None if idncaption == [] else \
                   {'type': '', 'info': '', 'licenses': None,
                    'annotations': idncaption,
                    'images': idnimage}

        createJson(idnprediction, val_data)
        return None

        


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
    parser.add_argument('--prop_steps', type=int , default=-1)
    parser.add_argument('--msm',type=str,default="ps",
            help='ps: probability score, ce: CrossEntropyLoss, co: cocoEval')
    parser.add_argument('--test_prop0', action='store_true')
    parser.add_argument('--test_c_step', action='store_true')
    parser.add_argument('--no_avg', action='store_true')
    parser.add_argument('--filepath', type=str , default='hint_improvement.pkl')
    parser.add_argument('--update_step', type=int , default=0)
    parser.add_argument('--update_method', type=str , default='c')
    parser.add_argument('--load_val', action='store_true',
            help='if true, loads gt for all samples into json file in data dir')
    args = parser.parse_args()
    print(args)
    main(args)
