import pickle
from data_loader import CocoDataset
from build_vocab import Vocabulary

with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)


coco = CocoDataset(root='./data/val_resized2014',
                   json='./data/annotations/captions_val2014.json',
                   vocab=vocab,
                   transform=None)

for i in range(100,200):
  print coco[i][0]
  print coco[i][1]