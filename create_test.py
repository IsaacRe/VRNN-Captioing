import pickle
from data_loader import CocoDataset
from build_vocab import Vocabulary

with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)


coco = CocoDataset(root='./data/resizedVal2014',
                   json='./data/annotations/captions_val2014.json',
                   vocab=vocab,
                   transform=None)

for i in range(0,3):
  print coco[i][0]
  print coco[i][1]