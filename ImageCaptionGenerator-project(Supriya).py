#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input,Dense,Dropout,Embedding,LSTM
from tensorflow.keras.layers import concatenate


# In[5]:


flickr_image="C:\Flickr8k\Flickr8k_dataset\Images"
flickr_token="C:\Flickr8k\Flickr8k_dataset\captions.txt"


# In[6]:


Images=os.listdir(flickr_image)
print("The number of jpg(images) flies in Flicker8k: {}".format(len(Images)))


# In[7]:


import random, os
path=("C:/Flickr8k/Flickr8k_dataset/Images/")
for i in range(4):
    img_name= random.choice([x for x in os.listdir(path)if os.path.isfile(os.path.join(path, x))])
    print(img_name)
    img = pltimg.imread(path+img_name)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis("on")
    plt.show()


# In[8]:


filename = (r'C:\Flickr8k\Flickr8k_dataset\captions.txt')

def load_descriptions(filename):
    file = open(filename)
    text = file.readlines()
    return text

# load descriptions 
text= load_descriptions(filename)

def image_to_captions(text):
    descriptions  = {}
    for line in text:
        token = line.split()
        image_id=token[0].split('.')[0] # separating with '.' to extract image id (removing .jpg)
        image_caption =' '.join(token[1:])
        
        if(image_id not in descriptions):
            descriptions[image_id] = [image_caption]
        else:
            descriptions[image_id].append(image_caption)
        
    return descriptions


# In[9]:


img_path=r'C:/Flickr8k/Flickr8k_dataset/Images/'
pic="478592803_f57cc9c461.jpg"
x=pltimg.imread(img_path+pic)
plt.imshow(x)

filename= r'C:\Flickr8k\Flickr8k_dataset\captions.txt'
text=load_descriptions(filename)
description=image_to_captions(text)
description['478592803_f57cc9c461']


# In[10]:


text= r'C:\Flickr8k\Flickr8k_dataset\captions.txt'
doc=load_descriptions(text)
description=image_to_captions(doc)
data=pd.DataFrame({'image_id':description.keys(),'caption':description.values()})
data.head(15)


# # Data Cleaning

# In[11]:


""" 1. lower each word
    2. remove puntuations
    3. remove words less than length 1 """ 

import string

def cleaning_text(descriptions):
    clean_descriptions={}
    
    #create a translation table,Remove all punctuation from each token.
    table = str.maketrans('', '', string.punctuation)
    
    for key in descriptions.keys():
        for idx in range(len(descriptions[key])):
            tokens= descriptions[key][idx].split()
            
            #remove punctuation from each token
            tokens=[token.translate(table) for token in tokens]
            
            #lowercase,remove hanging 's,remove tokens with numbers in them
            tokens=[token.lower() for token in tokens if len(token)>1 if token.isalpha()]
            descriptions[key][idx]=' '.join(tokens)
            
    return descriptions 


# ## Before the clean_description

# In[12]:


text= open(r'C:\Flickr8k\Flickr8k_dataset\captions.txt')
caption=text.read().split('\n')
caption[:6]


# ## After the clean_description

# In[13]:


text= r'C:\Flickr8k\Flickr8k_dataset\captions.txt'
doc=load_descriptions(text)
description=image_to_captions(doc)
clean_descriptions=cleaning_text(description)
clean_descriptions['1000268201_693b08cb0e']


# ## Build vocabulary of all unique words

# In[14]:


def create_vocabulary(clean_descriptions):
    vocabulary = set()
    for img_captions in clean_descriptions.values(): # list of 5 captions for each image
        for caption in img_captions:
            for token in caption.split():
                vocabulary.add(token)    
    return vocabulary

vocabulary=create_vocabulary(clean_descriptions)
print('vocabulary size',len(vocabulary))


# In[15]:


all_vocab=[]
for key in clean_descriptions.keys():
    for desc in clean_descriptions[key]:
        for i in desc.split():
            all_vocab.append(i)
            
print('All vocab size: %d' %len(all_vocab))
print(all_vocab[:15])


# ## save the captions

# In[16]:


def save_captions(clean_descriptions,filename):
    data = []
    for image_id,image_captions in clean_descriptions.items():
        for caption in image_captions:
            data.append(image_id + ' ' + caption + '\n')
            
    with open(filename,'w') as file:
        for line in data:
            file.write(line)


# In[17]:


save_captions(clean_descriptions,'descriptions.txt')


# ## count the repeated word

# In[18]:


from collections import Counter

text=open(r'C:\Flickr8k\Flickr8k_dataset\captions.txt')
caption=text.read().split()

count=Counter(caption)
Repeated_word_count= count.most_common(50)
print(Repeated_word_count)


# In[19]:


data=Repeated_word_count
df=pd.DataFrame(data,columns=['words','counts'])
df.head(50)


# ### Distribution of word and count

# In[20]:


import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(20,6))
sns.barplot(x='words',y='counts',data=df[:50])
plt.xticks(rotation='vertical')
plt.title("Plot of count vs words")


# In[21]:


data= Repeated_word_count
df=pd.DataFrame(data,columns=['word','count'])
import seaborn as sns
sns.set(style="darkgrid")
df['count'].plot(figsize=(20,8),color='blue')
sns.histplot(data=df[:50],x='word')
plt.xticks(rotation='vertical')
plt.title("plot of count vs word")


# In[22]:


data= Repeated_word_count
df= pd.Series(data[:50])
df.describe()


# # Preprocessing for images

# In[23]:


from tensorflow.keras.preprocessing.image import load_img,img_to_array

from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions


def preprocess_image(img_path):
    img = tensorflow.keras.preprocessing.image.load_img(img_path,target_size=(224,224)) 
    img = tensorflow.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    return img


# In[24]:


import tensorflow
plt.imshow(tensorflow.keras.preprocessing.image.load_img('C:/Flickr8k/Flickr8k_dataset/Images/367925122_335ed279a8.jpg',
                                                         target_size=(224,224)))


# ## To train the image_id

# In[25]:


def img_id_train(filename):
    with open(filename) as file:
        data = file.readlines()
        train_img_name = []
        for img_id in data:
            train_img_name.append(img_id.split('.')[0])
    return train_img_name 


# ## To train the image-caption

# In[26]:


def load_captions(filename, img_name):
    
    train_captions = {}    
    
    for line in doc:
        tokens = line.split()
        img_id, image_caption = tokens[0], tokens[2:-1]
        image_id=img_id.split('.')[0]
        modified_caption = '<BEGIN> ' + ' '.join(image_caption) + '<END>'
        
        if(image_id not in train_captions):
            train_captions[image_id] = []
        
            train_captions[image_id].append(modified_caption)
    
    return train_captions


# ## After the trained captions

# In[27]:


plt.imshow(tf.keras.preprocessing.image.load_img('C:/Flickr8k/Flickr8k_dataset/Images/110595925_f3395c8bd6.jpg',
                                                         target_size=(224,224)))
filename=open(r'C:\Flickr8k\Flickr8k_dataset\train_images.txt') 
doc=filename.readlines()
img_name=['110595925_f3395c8bd6']
train_captions= load_captions(doc, img_name)
train_captions['110595925_f3395c8bd6']


# ## Divide all captions into train, valid, test
# 

# In[28]:


train_img_txt = 'C:/Flickr8k/Flickr8k_dataset/train_images.txt'
valid_img_txt = 'C:/Flickr8k/Flickr8k_dataset/val_images.txt'
test_img_txt  = 'C:/Flickr8k/Flickr8k_dataset/testImages.txt'

train_img_name = img_id_train(train_img_txt)
valid_img_name = img_id_train(valid_img_txt )
test_img_name  = img_id_train(test_img_txt)

train_captions = load_captions('clean_descriptions.txt', train_img_name)
valid_captions = load_captions('clean_descriptions.txt', valid_img_name)
test_captions = load_captions('clean_descriptions.txt', test_img_name)


# In[29]:


#matplotlib.pyplot.subplot(nrows,ncol,index)
plt.subplot(1,3,1)
plt.imshow(tf.keras.preprocessing.image.load_img('C:/Flickr8k/Flickr8k_dataset/Images/109202801_c6381eef15.jpg',target_size=(224,224)))
print("Train-Caption:6000")
print(train_captions['109202801_c6381eef15'][0])
print()

plt.subplot(1,3,2)
plt.imshow(tf.keras.preprocessing.image.load_img('C:/Flickr8k/Flickr8k_dataset/Images/109202756_b97fcdc62c.jpg',target_size=(224,224)))
print("Valid-Caption:1000")
print(valid_captions['109202756_b97fcdc62c'][0])
print()

plt.subplot(1,3,3)
plt.imshow(tf.keras.preprocessing.image.load_img('C:/Flickr8k/Flickr8k_dataset/Images/109671650_f7bbc297fa.jpg',target_size=(224,224)))
print("Test-Caption:1000")
print(test_captions['109671650_f7bbc297fa'][0])
print()


# ## split caption Create vocabulary set (each vocabulary is unique)

# In[30]:


from collections import Counter
all_train_captions = []
for captions in train_captions.values():
    for caption in captions:
        all_train_captions.append(caption)
        
corpus = []
for caption in all_train_captions:
    for token in caption.split():
        corpus.append(token)


# # Count the number of times a word appears in a train caption.

# In[31]:


#count the repeat-word(vocab)       
descriptions = Counter(corpus)
print(len(descriptions))
word_count= descriptions.most_common(50)
print(word_count)

vocab = []
for token,count in descriptions.items():
    if(count>=1):
        vocab.append(token)


# # Plot of count vs word

# In[32]:


data= word_count
df=pd.DataFrame(data,columns=['word','count'])
import seaborn as sns
sns.set(style="darkgrid")
df['count'].plot(figsize=(20,6),color='blue')
sns.barplot(x='word',y='count',data=df[:50])
plt.xticks(rotation='vertical')
plt.title("plot of count vs word")

