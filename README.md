# Image Captioning Auto-Completion

The goal of image captioning is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). In this tutorial, we used [resnet-152](https://arxiv.org/abs/1512.03385) model pretrained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) image classification dataset. The decoder is a long short-term memory (LSTM) network. 

![alt text](png/model.png)

#### Training phase
For the encoder part, the pretrained CNN extracts the feature vector from a given input image. The feature vector is linearly transformed to have the same dimension as the input dimension of the LSTM network. For the decoder part, source and target texts are predefined. For example, if the image description is **"Giraffes standing next to each other"**, the source sequence is a list containing **['\<start\>', 'Giraffes', 'standing', 'next', 'to', 'each', 'other']** and the target sequence is a list containing **['Giraffes', 'standing', 'next', 'to', 'each', 'other', '\<end\>']**. Using these source and target sequences and the feature vector, the LSTM decoder is trained as a language model conditioned on the feature vector.

#### Test phase
In the test phase, the encoder part is almost same as the training phase. The only difference is that batchnorm layer uses moving average and variance instead of mini-batch statistics. This can be easily implemented using [encoder.eval()](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/sample.py#L41). For the decoder part, there is a significant difference between the training phase and the test phase. In the test phase, the LSTM decoder can't see the image description. To deal with this problem, the LSTM decoder feeds back the previosly generated word to the next input. This can be implemented using a [for-loop](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py#L57-L68).



## Usage 


#### 1. Clone the repositories
```bash
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install
$ cd ../../
$ git clone https://github.com/tylin/coco-caption.git
$ git clone https://github.com/brucepang/image_captioning_autocompletion.git
$ cd image_captioning_autocompletion/
```
This code is written in Python2.7 and requires [Pytorch 0.3.0](https://pytorch.org/previous-versions/).

Note: need to comment SPICE score out from eval.py in coco-caption/pycocoevalcap. It is still not stable yet.

#### 2. Download the dataset

```bash
$ pip install -r requirements.txt
$ chmod +x download.sh
$ ./download.sh
```

#### 3. Preprocessing

```bash
$ python utils/build_vocab.py   
$ python utils/resize.py
```

#### 4. Train the model

```bash
$ python train.py    
```

#### 5. Evaluate the model 

We have provided a evaluate script to evaluate the model using multiple scorer like bleu, meteor, and rouge. Check the file to see the full list of accepted arguments.
```bash
$ python test/eval.py --c_step --num_samples --num_hints ...

```
We have provided a web application written in Flask framework that help visualize the captioning.
#### 6. Select testing images for the web application
```bash
$ python utils/create_test.py
$ cd application/static/candidate
$ mkdir image_candidate
$ mkdir caption_candidate
$ mv *.jpg image_candidate/
$ mv *.txt image_candidate/

```

#### 7. Setup local database for the application
```bash
$ cd ../../
$ python database_setup.py
$ python database_initialize.py

```

#### 8. Test the model using the web application
```bash
$ python application_step1.py
It will be on port 5000. If you are running the application in remote server, try the following ssh command.
$ ssh -f yourusername@servername -L anylocalport:localhost:5000 -N
Then go to your favorite browser and type http://localhost:anylocalport/.

```

#### 9. Test the model with simple script
We have provided a simple sampling script step_1.py in test/ which will simulate the process of feeding user inputs one by one and see the new translation. Check the file to see the full list of accepted arguments.

```bash
python test/step_1.py --image 1.

```
<br>

## Pretrained model
If you do not want to train the model from scratch, you can use a pretrained model. You can download the pretrained model [here](https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0) and the vocabulary file [here](https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip?dl=0). You should extract pretrained_model.zip to `./models/` and vocab.pkl to `./data/` using `unzip` command.
