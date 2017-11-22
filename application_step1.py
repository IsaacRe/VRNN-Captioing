# Author: Bo Pang
# Date : Nov 22 2017

from flask import Flask, render_template, request, redirect,jsonify, url_for, flash
import datetime
from sqlalchemy import create_engine, asc
from sqlalchemy.orm import sessionmaker
from database_setup import Base, Input
from flask import session as login_session
import random, string
# import httplib2
import json
from flask import make_response
import requests
from build_vocab import Vocabulary
import pickle
from step_1 import encode, decode
from model import EncoderCNN, DecoderRNN
import torch

app = Flask(__name__)


APPLICATION_NAME = "Image Captioning Autocompletion"
gv = {}

#Connect to Database and create database session
engine = create_engine('sqlite:///course.db')
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()

# load vocabulary
with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

decoder = DecoderRNN(256, 512,len(vocab), 1)
if torch.cuda.is_available():
    decoder.cuda()
decoder.load_state_dict(torch.load('./models/decoder-4-3000.pkl'))

#Show all courses with all instructors
@app.route('/')
@app.route('/demo')
def showDemos():
  return render_template('demos.html')


#Show a course's all instructors
@app.route('/demo/step1')
def showStep1():
    image = session.query(Input).filter_by(id = 1).one()
    return render_template('step1.html',image=image)

@app.route('/demo/step1/test',methods=['GET','POST'])
def extractFeature():
    if request.method == 'GET':
        image = session.query(Input).filter_by(id = 1).one()
        feature = encode("."+image.name,vocab)
        gv["feature"]=feature
        sentence = decode(feature,[],decoder,vocab)
        print sentence
        image.translation = sentence
        return render_template('caption.html',image=image)
    else:
        if request.form['hint']:
            hints = []
            hints.append(vocab.word2idx["<start>"])
            image = session.query(Input).filter_by(id = 1).one()
            for word in (request.form['hint']).split():
                hints.append(vocab.word2idx[word])
            sentence = decode(gv["feature"],hints,decoder,vocab)
            print sentence
            image.translation = sentence
            return render_template('caption.html',image=image)


if __name__ == '__main__':
  app.secret_key = 'super_secret_key'
  app.debug = True
  app.run(host = '0.0.0.0', port = 5000)
