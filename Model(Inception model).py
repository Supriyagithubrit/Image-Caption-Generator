#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import argmax
import pandas as pd 
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

import nltk
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import Adam, RMSprop
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from os import listdir
from collections import Counter
from tensorflow.keras.layers import concatenate
from os import listdir


# ## Extract the feature Vector from all images

# InceptionV3 is a CNN that is 48 layers, input shape has to be (299,299,3) and loaded with weights pre-trained on ImageNet.

# In[2]:


model=InceptionV3(weights = 'imagenet')
new_model= Model(model.input,model.layers[-2].output)
#new_model.summary()


# ## Preprocessing for images

# In[8]:


import tensorflow
img_path=('C:/Flickr8k/Flickr8k_dataset/Images/')
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

def preprocess_image(img_path):
    img = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=(299,299,3)) 
    img = tensorflow.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    return img


# In[9]:


def images_name(img_path):
    img_name = set([img_path+image for image in listdir(img_path)])
    return img_name


# Encode image: 
# Autoencoders are a deep learning model for transforming data from a high-dimensional space to a lower-dimensional space. They work by encoding the data, whatever its size, to a 1-D vector. This vector can then be decoded to reconstruct the original data.

# In[10]:


# Function to encode given image into a vector of size (2048, )
def encode_image(image,new_model):
    image = preprocess_image(image)
    feature_vector = new_model.predict(image)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1]) # reshape from (1, 2048) to (2048, )
    return feature_vector


# In[11]:


image="C:/Flickr8k/Flickr8k_dataset/kid.jpg"
encode_img=encode_image(image,new_model)
print(encode_img)
print(encode_img.shape)


# ## To train the image-id,image-caption

# In[13]:


def img_id_train(filename):
    with open(filename) as file:
        data = file.readlines()
        train_img_name = []
        for img_id in data:
            train_img_name.append(img_id.split('.')[0])
    return train_img_name 


# ### calculate the length of images 

# In[87]:


filename=r'C:/Flickr8k/Flickr8k_dataset/train_images.txt'
train_img_name=img_id_train(filename)
print('Length of train images=',len(train_img_name))


# In[14]:


filename=r'C:/Flickr8k/Flickr8k_dataset/train_images.txt'
def load_captions(filename, img_name):

    train_captions = {}    
    
    for line in doc:
        tokens = line.split()
        img_id, image_caption = tokens[0], tokens[2:-1]
        image_id=img_id.split('.')[0]
        modified_caption = '<BEGIN> ' + ' '.join(image_caption) + '<END>'
        
        if(image_id not in train_captions):
            train_captions[image_id] = []
        else:
            train_captions[image_id].append(modified_caption)
    
    return train_captions


# ## Divide all caption into train ,valid ,test

# Train_captions

# In[15]:


filename=open(r'C:\Flickr8k\Flickr8k_dataset\train_images.txt') 
doc=filename.readlines()
img_name=['110595925_f3395c8bd6']
train_captions= load_captions(doc, img_name)
train_captions['110595925_f3395c8bd6']


# Valid_captions

# In[16]:


valid_img_txt = 'C:/Flickr8k/Flickr8k_dataset/val_images.txt'
valid_img_name = img_id_train(valid_img_txt )
valid_captions = load_captions('descriptions.txt', valid_img_name)
valid_captions['47871819_db55ac4699']


# Test_captions

# In[17]:


test_img_txt  = 'C:/Flickr8k/Flickr8k_dataset/testImages.txt'
test_img_name  = img_id_train(test_img_txt)
test_captions = load_captions('descriptions.txt', test_img_name)
test_captions['118187095_d422383c81']


# In[92]:


import pickle
train_features =open("C:\Flickr8k\Flickr8k_dataset\encoded_train_images.pkl", "rb") 
train_feature = pickle.load(train_features)
valid_features= open("C:\Flickr8k\Flickr8k_dataset\encoded_valid_images.pkl", "rb") 
valid_features = pickle.load(valid_features)
test_features= open("C:\Flickr8k\Flickr8k_dataset\encoded_test_images.pkl", "rb") 
test_features = pickle.load(test_features)


# In[93]:


from collections import Counter
all_train_captions = []
for captions in train_captions.values():
    for caption in captions:
        all_train_captions.append(caption)
        
corpus = []
for caption in all_train_captions:
    for token in caption.split():
        corpus.append(token)
        
#count the repeat-word(vocab)       
descriptions = Counter(corpus)
print(len(descriptions))
word_count= descriptions.most_common(50)
print(word_count)

vocab = []
for token,count in descriptions.items():
    if(count>=1):
        vocab.append(token)


# ## calculate maximum length of caption to decide the model structure parameters.
# 

# In[94]:


def max_len_caption(all_train_captions):   
    max_len = 0
    for caption in all_train_captions:
        max_len = max(max_len,len(caption.split()))
    print('Maximum length of caption= ',max_len)


# In[95]:


max_length_caption = max_len_caption(all_train_captions)


# ### Read the pre-trained Embedding weights and create the Embedding matrix

# In[96]:


word_to_index = {}
index_to_word = {}
    
for idx,token in enumerate(vocab):
    word_to_index[token] = idx+1
    index_to_word[idx+1] = token

vocab_size = len(index_to_word) + 1

#The size of our vocabulary is 8175 words.
vocab_size


# In[97]:


embeddings_index = {}

#Glove-Global-vectors-for-word-representation or "GloVe" is an unsupervised learning algorithm for obtaining 
#vector representations for words
file = open('C:/Flickr8k/Flickr8k_dataset/glove.6B.200d.txt', encoding="utf-8")

for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
file.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector
        
print("Embedding_matrix.shape = ",embedding_matrix.shape)


# In[98]:


print(embedding_matrix)


# In[99]:


word_to_idx=pickle.load(open('word_to_idx.pkl','rb'))
print(len(word_to_idx))
# word_to_idx


# In[100]:


idx_to_word=pickle.load(open('idx_to_word.pkl','rb'))
print(len(idx_to_word))
# idx_to_word


# # Create a Data generator
# 

# In[101]:


# data generator, intended to be used in a call to model.fit_generator()
#num_photos_per_batch

def data_generator(descriptions, photos, word_to_idx, max_length_caption,  BATCH_SIZE):
    X1, X2, y = list(), list(), list()
    n=0
    
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            
            temp=r'C:/Flickr8k/Flickr8k_dataset/Images/'
            
            photo = photos[temp + key +'.jpg']
            
            # move through each description for the image
            for desc in desc_list:
                
                # encode the sequence
                seq = [word_to_idx[word] for word in desc.split(' ') if word in word_to_idx]
                
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                   
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                  
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0


# In[102]:


train_generator = data_generator(train_captions, train_features, word_to_index, max_length_caption=39, BATCH_SIZE=5)
valid_generator = data_generator(valid_captions, valid_features, word_to_index, max_length_caption=39, BATCH_SIZE=5)
test_generator = data_generator(test_captions, test_features, word_to_index, max_length_caption=39, BATCH_SIZE=5)
train_generator


# In[103]:


def greedySearch(photo, model, max_length_caption, word_to_idx, idx_to_word):
    
    in_text = 'startseq'
    for i in range(max_length_caption):
        sequence = [word_to_ix[w] for w in in_text.split() if w in word_to_ix]
        sequence = pad_sequences([sequence], maxlen=max_length_caption)
        y = model.predict([photo,sequence], verbose=0)
        y = np.argmax(y)
        word = idx_to_word[y]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# ## Model Architecture

# In[113]:


from tensorflow.keras.utils import plot_model
# define the captioning model

def define_model(max_length,vocab_size):
   
    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)   #fe1:feature 
    fe2 = Dense(256, activation='relu')(fe1)
    
    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=False)(inputs2) # se1:sequence
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model


# # Create image_captioning model

# In[114]:


model_ic = define_model(max_length_caption, vocab_size)


# In[115]:


model_ic.summary()


# In[116]:


plot_model(model_ic, to_file='model_ic.png', show_shapes=True)


# In[117]:


print(model.layers[2].input)
print(model.layers[2].output)
#model.layers[2].set_weights(embedding_matrix)
model.layers[2].trainable = False


# In[118]:


import keras 
import tensorflow
from tensorflow.keras.metrics import CosineSimilarity,Precision
LOSS = 'categorical_crossentropy'
OPTIM = 'adam'
METRICS = [tensorflow.keras.metrics.CosineSimilarity(),
           tensorflow.keras.metrics.Precision()]


# In[136]:


model.compile(loss=LOSS, optimizer=OPTIM, metrics=METRICS)
#model.summary()


# In[120]:


BATCH_SIZE = 5
EPOCHS = 20
STEPS = len(train_captions)//BATCH_SIZE
VALID_STEPS = len(valid_captions)//BATCH_SIZE

print(STEPS)
print(VALID_STEPS)


# # Train the model

# In[135]:


history = model.fit(train_generator, 
                    epochs=EPOCHS, 
                    steps_per_epoch=STEPS, 
                    verbose=1,
                    validation_data=valid_generator,
                    validation_steps=VALID_STEPS)


# In[ ]:


def show_history_metrics(history, metrics_name=None):
    if metrics_name==None:
         print("No performance metrics specified")
    else:
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(history.history[metrics_name])
        plt.plot(history.history['val_'+metrics_name])
        plt.title('Model '+metrics_name)
        plt.ylabel(metrics_name)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.show()


# In[99]:


show_history_metrics(history, 'loss')
show_history_metrics(history, 'precision_1')


# In[100]:


scores = model_ic.evaluate(test_generator, steps=VALID_STEPS)
print(scores)
print("Loss:      %.5f" % scores[0])
print("Precision: %.5f" % scores[2])


# In[101]:


def Predict_test_caption(test_image_id, model, show_predict=True, CNN_units_num=2048):
    print("image_id:"+test_image_id)
    test_image_filename = 'C:/Flickr8k/Flickr8k_dataset/Images/'+test_image_id+'.jpg'
    image = encoding_test[test_image_filename].reshape((1,CNN_units_num))
    pred_caption = greedySearch(image, model, max_length_caption, word_to_index, index_to_word)
    true_captions = preprocessed_map[test_image_id]
    if show_predict:
        x=plt.imread(test_image_filename)
        plt.imshow(x)
        plt.show()
        print("Predict:\n"+pred_caption)
        print("True:")
        print(*preprocessed_map[test_image_id],sep='\n')
    return pred_caption, true_captions


# In[102]:


def compute_BLEU(pred_caption, true_captions, show_bleu=True): 
    bleu = [0.0, 0.0, 0.0, 0.0]
    references = [true_captions[0].split(),true_captions[1].split(),true_captions[2].split(),true_captions[3].split(),true_captions[4].split()]
    hypothesis = pred_caption.split()
    smooth = SmoothingFunction()
    bleu[0] = sentence_bleu(references, hypothesis, weights=(1.0, 0, 0, 0), smoothing_function=smooth.method1)
    bleu[1] = sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
    bleu[2] = sentence_bleu(references, hypothesis, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smooth.method1)
    bleu[3] = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    if show_bleu:
        print('BLEU-1: %f' % bleu[0])
        print('BLEU-2: %f' % bleu[1])
        print('BLEU-3: %f' % bleu[2])
        print('BLEU-4: %f' % bleu[3])
        
    return bleu


# In[103]:


def evaluate_BLEU(encoding_features, images_name, model, show_results=True, CNN_units_num=2048):
    mean_bleu = np.zeros(4)
    for test_id in iter(images_name):
        test_image_filename = 'C:/Flickr8k/Flickr8k_dataset/Images/'+test_id+'.jpg'
        image = encoding_test[test_image_filename].reshape((1,CNN_units_num))
        pred_caption = greedySearch(image, model, max_length_caption, word_to_index, index_to_word)
        true_captions = preprocessed_map[test_id]
         
        bleu = compute_BLEU(pred_caption, true_captions, show_bleu=False)
        bleu_temp = np.array(bleu)
        mean_bleu = mean_bleu + bleu_temp
    
    mean_bleu = mean_bleu/len(images_name)
    if show_results:
        print('MEAN_BLEU-1: %f' % mean_bleu[0])
        print('MEAN_BLEU-2: %f' % mean_bleu[1])
        print('MEAN_BLEU-3: %f' % mean_bleu[2])
        print('MEAN_BLEU-4: %f' % mean_bleu[3])
    return mean_bleu    


# In[104]:


with open("encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = pickle.load(encoded_pickle)


# 14.Model Average Sentence_BLEU

# In[105]:


Mean_bleu = evaluate_BLEU(encoding_test, test_img_name, model_ic)
C_bleu = evaluate_corpursBLEU(encoding_test, test_img_name, model_ic)


# 15.Preview the prediction result, enter the image id

# In[107]:


pc, tc = Predict_test_caption('3605676864_0fb491267e', model_ic)
bleu = compute_BLEU(pc, tc)

