import os
from PIL import Image
import pickle
from data_loader import CocoDataset
from build_vocab import Vocabulary

with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


coco = CocoDataset(root='./data/val_resized2014',
                   json='./data/annotations/captions_val2014.json',
                   vocab=vocab,
                   transform=None)
output_dir = './application/static/candidate/'
for i in range(20, 40):
    img = coco[i][0]
    img.save(os.path.join(output_dir, str(i)+".jpg"), img.format)
    with open(output_dir+str(i)+'.txt', 'w') as f:
        caption = ' '.join([vocab.idx2word[id] for id in coco[i][1][1:-1]])
        print caption
        f.write(caption)
