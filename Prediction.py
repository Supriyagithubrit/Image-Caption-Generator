#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input,Dense,Dropout,Embedding,LSTM
from tensorflow.keras.layers import concatenate


# In[2]:


import model


# # Predict test caption

# In[3]:


filename = (r'C:\Flickr8k\Flickr8k_dataset\captions.txt')
photo=("C:/Flickr8k/Flickr8k_dataset/Images/")
word_to_idx=pickle.load(open('word_to_idx.pkl','rb'))
maxlen=35


def cap(photo):
    in_text = '<Begin>'
    
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word
        
        if word == '<End>':
            break
        
        
    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


# In[4]:


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


# In[5]:


with open("C:\Flickr8k\Flickr8k_dataset\encoded_test_images.pkl", "rb") as encoded_test_pickle:
    encoding_test = pickle.load(encoded_test_pickle)
    images=("C:/Flickr8k/Flickr8k_dataset/Images/")

for i in range(1):
    random = np.random.randint(0,1)
    img_name = list(encoding_test.keys())[random]
    print(img_name)
  
    photo = encoding_test[img_name].reshape((1,2048))
    
    i = pltimg.imread(images+img_name)
    plt.imshow(i)
    plt.axis("on")
    plt.show()

    filename = (r'C:\Flickr8k\Flickr8k_dataset\captions.txt')
    text= load_descriptions(filename)
    predict_captions= predict_caption(text)
    print(predict_captions['1056338697_4f7d7ce270'])
    print('\n')


# In[6]:


pic="2061144717_5b3a1864f0.jpg"
x=pltimg.imread(images+pic)
plt.imshow(x)
print('\n')
text= load_descriptions(filename)
predict_captions= predict_caption(text)
print(predict_captions['2061144717_5b3a1864f0'])


# In[8]:


pic="3044746136_8b89da5f40.jpg"
x=pltimg.imread(images+pic)
plt.imshow(x)
print('\n')
text= load_descriptions(filename)
predict_captions= predict_caption(text)
predict_captions['3044746136_8b89da5f40']


# In[9]:


pic="2453971388_76616b6a82.jpg"
x=pltimg.imread(images+pic)
plt.imshow(x)
text= load_descriptions(filename)
predict_captions= predict_caption(text)
print(predict_captions['2453971388_76616b6a82'])


# In[10]:


pic="2458269558_277012780d.jpg"
x=pltimg.imread(images+pic)
plt.imshow(x)
text= load_descriptions(filename)
predict_captions= predict_caption(text)
print(predict_captions['2458269558_277012780d'])


# In[11]:


pic="2182488373_df73c7cc09.jpg"
x=pltimg.imread(images+pic)
plt.subplot(1,1,1)
plt.imshow(x)
text= load_descriptions(filename)
predict_captions= predict_caption(text)
print(predict_captions['2182488373_df73c7cc09'])


# In[12]:


pic="270816949_ffad112278.jpg"
x=pltimg.imread(images+pic)
plt.imshow(x)
text= load_descriptions(filename)
predict_captions= predict_caption(text)
print(predict_captions['270816949_ffad112278'])


# In[13]:


pic="211295363_49010ca38d.jpg"
x=pltimg.imread(images+pic)
plt.imshow(x)
text= load_descriptions(filename)
predict_captions= predict_caption(text)
print(predict_captions['211295363_49010ca38d'])


# In[14]:


pic="1096395242_fc69f0ae5a.jpg"
x=pltimg.imread(images+pic)
plt.imshow(x)
text= load_descriptions(filename)
predict_captions= predict_caption(text)
print(predict_captions['1096395242_fc69f0ae5a'])


# In[15]:


pic="1131800850_89c7ffd477.jpg"
x=pltimg.imread(images+pic)
plt.imshow(x)
text= load_descriptions(filename)
predict_captions= predict_caption(text)
print(predict_captions['1131800850_89c7ffd477'])


# In[16]:


pic="1119015538_e8e796281e.jpg"
x=pltimg.imread(images+pic)
plt.imshow(x)
text= load_descriptions(filename)
predict_captions= predict_caption(text)
print(predict_captions['1119015538_e8e796281e'])


# In[ ]:





# In[ ]:




