#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout,MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from keras.applications.imagenet_utils import decode_predictions 
from sklearn import preprocessing
import glob
import cv2
import os
import gc

from sklearn import svm

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the dataset

# In[2]:


train_data = []
train_labels = [] 


# In[3]:


for directory_path in glob.glob("dataset_central/training/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  
        img = cv2.resize(img, (508, 274))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_data.append(img)
        train_labels.append(label)
        


# In[4]:


len(train_labels)


# In[5]:


train_data = np.array(train_data)
train_labels = np.array(train_labels)


# In[6]:


validation_data = []
validation_labels = [] 


# In[7]:


for directory_path in glob.glob("dataset_central/validation/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (508, 274))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        validation_data.append(img)
        validation_labels.append(label)


# In[8]:


validation_data = np.array(validation_data)
validation_labels = np.array(validation_labels)


# # Transform the labels

# In[9]:


le = preprocessing.LabelEncoder()


# In[10]:


le.fit(validation_labels)
y_test = le.transform(validation_labels)


# In[11]:


le.fit(train_labels)
y_train = le.transform(train_labels)


# In[12]:


train_data.shape


# In[13]:


train_data = train_data.reshape(train_data.shape[0],train_data.shape[2],train_data.shape[1],train_data.shape[3])


# In[14]:


validation_data = validation_data.reshape(validation_data.shape[0],validation_data.shape[2],validation_data.shape[1],validation_data.shape[3])


# # Feature extraction

# In[15]:


resnet_model = Sequential()


resnet_model = tf.keras.applications.ResNet50(include_top=False,

                   input_shape=(508, 274,3),

                   pooling='avg',

                   weights='imagenet')

for each_layer in resnet_model.layers:

        each_layer.trainable=False
        


# In[16]:


print(resnet_model.summary())


# In[17]:


X_train = resnet_model.predict(train_data)


# In[18]:


X_test = resnet_model.predict(validation_data)


# # Classification

# In[19]:


svm_model = svm.SVC(C=1, gamma=1, kernel='linear')


# In[20]:


svm_model.fit(X_train,y_train)


# # Model evaluation

# In[21]:


svm_model.score(X_test,y_test)


# In[22]:


svm_model.score(X_train,y_train)

