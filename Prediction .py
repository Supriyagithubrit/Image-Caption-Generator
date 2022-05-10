#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from numpy import argmax
import pandas as pd 
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

import pickle
from pickle import load,dump
import keras
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions

from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input,Dense,Dropout,Embedding,LSTM
from tensorflow.keras.layers import concatenate


# # predict Caption

# In[13]:


filename = (r'C:\Flickr8k\Flickr8k_dataset\captions.txt')

def load_descriptions(filename):
    file = open(filename)
    text = file.readlines()
    return text

# load descriptions 
text= load_descriptions(filename)

def predict_caption(text):
# creating a "descriptions" dictionary  where key is 'img_id' and value is list of captions corresponding to that image_file.
    predict_caption = {}
 
    for img_to_cap in text:
        captions= img_to_cap.split()
        image_id=captions[0].split(".")[0]# separating with '.' to extract image id
        image_caption= captions[1:]
        modified_caption = '<Begin>' + ' '.join(image_caption) + '<End>'
      
        if(predict_caption.get(image_id)== None):
            predict_caption[image_id] = [modified_caption]
 
    return predict_caption


# # Predictions

# In[14]:


filename = (r'C:\Flickr8k\Flickr8k_dataset\captions.txt')
text= load_descriptions(filename)
photo=("C:/Flickr8k/Flickr8k_dataset/Images/")
word_to_idx=pickle.load(open('word_to_idx.pkl','rb'))
maxlen=35


def predict_cap(photo):
    in_text = "startseq"
    
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word
        
        if word =='endseq':
            break
        
        
    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


# In[53]:


with open("C:\Flickr8k\Flickr8k_dataset\storage\encoded_test_images.pkl", "rb") as encoded_test_pickle:
    encoding_test = pickle.load(encoded_test_pickle)
    images=("C:/Flickr8k/Flickr8k_dataset/Images/")

for i in range(1):
   # random = np.random.randint(0,1000)
    img_name = list(encoding_test.keys())[120]
    print(img_name)
  
    photo = encoding_test[img_name].reshape((1,2048))
    
    i = pltimg.imread(images+img_name)
    plt.imshow(i)
    plt.axis("on")
    plt.show()

    filename = (r'C:\Flickr8k\Flickr8k_dataset\captions.txt')
    text= load_descriptions(filename)
    predict_captions= predict_caption(text)
    print(predict_captions['2078311270_f01c9eaf4c'])
    print('\n')


# # Data cleaning

# In[ ]:


""" 1. lower each word
    2. remove puntuations
    3. remove words less than length 1 """ 

import string

def cleaning_text(descriptions):
    clean_descriptions= {}
    
    #create a translation table,Remove all punctuation from each token.
    table=str.maketrans('','',string.punctuation)
    
    for key in descriptions.keys():
        for idx in range(len(descriptions[key])):
            tokens = descriptions[key][idx].split()
            
            #remove punctuation from each token
            tokens= [token.translate(table) for token in tokens]
            
            #lowercase,remove hanging 's ,remove tokens with numbers in them
            tokens = [token.lower() for token in tokens if len(token)>1 if token.isalpha()]
            descriptions[key][idx] = ' '.join(tokens)
            
    return clean_descriptions
    


# In[ ]:


token = (r'C:\Flickr8k\Flickr8k_dataset\captions.txt')
text= load_descriptions(token)
clean_descriptions=image_to_caption(text)
clean_descriptions['1000268201_693b08cb0e']


# # Build vocabulary of all unique words

# In[ ]:


def create_vocabulary(clean_descriptions):

    #build vocabulary of all unique words
    vocabulary=set()
    for img_caption in clean_descriptions.keys():
        for desc in clean_descriptions[img_caption]:
            vocabulary.update(desc.split())
    
    return vocabulary 

vocabulary=create_vocabulary(clean_descriptions)
print('Vocabulary size',len(vocabulary))


# In[ ]:


all_vocab=[]
for key in clean_descriptions.keys():
     for desc in clean_descriptions[key]:
        for i in desc.split():
            all_vocab.append(i)
           
print('All vocab size: %d' % len(all_vocab))
print(all_vocab[:15])     


# # save descriptions

# In[ ]:


def save_descriptions(clean_descriptions,filename):
    data = []
    for image_id,image_captions in clean_descriptions.items():
        for caption in image_captions:
            data.append(image_id + ' ' + caption + '\n')
            
    with open(filename,'w') as file:
        for line in data:
            file.write(line)


# In[ ]:


save_descriptions(clean_descriptions,'descriptions.txt')


# # Training data

# In[ ]:


def train_Image_id(filename):
        file=open(filename)
        data = file.readlines()
        train_img_id = []
        for img_id in data:
            train_img_id.append(img_id.split('.')[0])
        return train_img_id


# In[ ]:


train_img_id=train_Image_id(filename)
train_img_id


# In[ ]:


word_to_idx=pickle.load(open('word_to_idx.pkl','rb'))
print(len(word_to_idx))


# In[ ]:


word_to_idx 


# In[ ]:




