# Author: Bo Pang
# Date : Nov 22 2017
import sys
from flask import Flask, make_response, render_template, request, redirect,jsonify, url_for, flash
import datetime
from sqlalchemy import create_engine, asc
from sqlalchemy.orm import sessionmaker
from application.database_setup import Base, Input
from flask import session as login_session
import random, string
import json
import requests
from utils.build_vocab import Vocabulary
import pickle
from step_1 import encode, decode, decode_word
from model import EncoderCNN, DecoderRNN
import torch
from evaluate import CocoJson, cocoEval
from utils import data_loader

app = Flask(__name__)


APPLICATION_NAME = "Image Captioning Autocompletion"
coco_json_dir = "./application/step_1/"
gv = {}

#Connect to Database and create database session
engine = create_engine('sqlite:///course.db')
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()

# load vocabulary
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

decoder = DecoderRNN(256, 512,len(vocab), 1)
if torch.cuda.is_available():
    decoder.cuda()
decoder.load_state_dict(torch.load('models/decoder_pretrained.pkl'))

#Show all courses with all instructors
@app.route('/')
@app.route('/demo')
def showDemos():
    return render_template('demos.html')


#Show a course's all instructors
@app.route('/demo/step1')
def selectImage():
    images = session.query(Input).order_by(asc(Input.id))
    return render_template("selectImage.html",images = images)

@app.route('/demo/step1/<path:image_id>')
def showStep1(image_id):
    print image_id
    image = session.query(Input).filter_by(id = image_id).one()
    return render_template('step1.html',image=image)


@app.route('/demo/nextword')
def showNextWord():
    image = session.query(Input).filter_by(id = 2).one()
    return render_template('step1_nextword.html',image=image)

@app.route('/demo/step1/test/<path:image_id>',methods=['GET','POST'])
def extractFeature(image_id):
    if request.method == 'GET':
        image = session.query(Input).filter_by(id = image_id).one()
        feature = encode("."+image.path,vocab)
        gv["feature"]=feature
        sentence_wo_update,sentence_with_update = decode(feature,[vocab.word2idx["<start>"]],decoder,vocab,c_step=5.0)
        print sentence_wo_update
        image.translation_noUpdate = sentence_wo_update

        coco_json = CocoJson(coco_json_dir+'gt.json', coco_json_dir+'pred.json')
        coco_json.add_entry(caption=image.ground_true, pred_caption=image.translation_noUpdate)
        coco_json.create_json()
        image.evalcap_noUpdate = cocoEval(coco_json.gt_json, coco_json.pred_json)
        
        print sentence_with_update
        image.translation_Update = sentence_with_update

        coco_json = CocoJson(coco_json_dir+'gt.json', coco_json_dir+'pred_update.json')
        coco_json.add_entry(pred_caption=image.translation_Update)
        coco_json.create_json()
        image.evalcap_Update = cocoEval(coco_json.gt_json, coco_json.pred_json)

        return render_template('caption.html',image=image, hint_word="")
    else:
        if request.form['hint']:
            hints = []
            hint_word = []
            hints.append(vocab.word2idx["<start>"])
            hint_word = ["<start>"]
            image = session.query(Input).filter_by(id = image_id).one()
            for word in (request.form['hint']).split():
                hints.append(vocab.word2idx[word.lower()])
                hint_word.append(word)
            sentence_wo_update,sentence_with_update = decode(gv["feature"],hints,decoder,vocab,c_step=5.0)
            sentence_wo_update = sentence_wo_update.split()
            image.translation_noUpdate = " ".join(sentence_wo_update[len(hint_word):])
            print image.translation_noUpdate
            
            coco_json = CocoJson(coco_json_dir+'gt.json', coco_json_dir+'pred.json')
            coco_json.add_entry(pred_caption=" ".join(hint_word)+" "+image.translation_noUpdate)
            coco_json.create_json()
            image.evalcap_noUpdate = cocoEval(coco_json.gt_json, coco_json.pred_json)

            sentence_with_update = sentence_with_update.split()
            image.translation_Update = " ".join(sentence_with_update[len(hint_word):])
            print image.translation_Update
            
            coco_json = CocoJson(coco_json_dir+'gt.json', coco_json_dir+'pred_update.json')
            coco_json.add_entry(pred_caption=" ".join(hint_word)+" "+image.translation_Update)
            coco_json.create_json()
            image.evalcap_Update = cocoEval(coco_json.gt_json, coco_json.pred_json)

            return render_template('caption.html',image=image,hint_word=" ".join(hint_word)+" ")


@app.route('/_find_next_word')
def findnword():
    print "findnword"
    hints=[]
    sentence = request.args.get('sentence', 0, type=str)
    for word in sentence.split():
        hints.append(vocab.word2idx[word.lower()])
    return jsonify(next_word=(decode_word(gv["feature"],hints,decoder,vocab)))

if __name__ == '__main__':
  app.secret_key = 'super_secret_key'
  app.debug = True
  app.run(host = '0.0.0.0', port = 5000)
