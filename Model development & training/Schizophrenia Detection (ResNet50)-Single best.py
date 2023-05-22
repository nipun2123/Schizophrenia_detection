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

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the dataset

# In[2]:


train_data = tf.keras.utils.image_dataset_from_directory(
    directory='dataset/training',
    labels='inferred',
    label_mode='binary',
    batch_size=32,
    class_names=["sc","healthy"],
    image_size=(508, 274))


# In[3]:


val_data = tf.keras.utils.image_dataset_from_directory(
    directory='dataset/validation',
    labels='inferred',
    label_mode='binary',
    batch_size=32,
    class_names=["sc","healthy"],
    image_size=(508, 274))


# # Model building and training

# In[4]:


resnet_model = Sequential()


pretrained_model = tf.keras.applications.ResNet50(include_top=False,

                   input_shape=(508, 274,3),

                   pooling='avg',

                   weights='imagenet')

for each_layer in pretrained_model.layers:

        each_layer.trainable=False
        

resnet_model.add(pretrained_model)


# In[5]:


resnet_model.add(Flatten())

resnet_model.add(Dense(512, activation='relu'))

resnet_model.add(Dense(1, activation='sigmoid'))


# In[6]:


resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
print(resnet_model.summary())


# In[7]:


history = resnet_model.fit(train_data, validation_data=val_data, epochs=30)


# # Model evaluation

# In[9]:


plt.figure(figsize=(8, 8))

epochs_range= range(30)

plt.plot( epochs_range, history.history['accuracy'], label="Training Accuracy")

plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")

plt.axis(ymin=0.4,ymax=1)

plt.grid()

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'validation'])


# In[10]:


plt.figure(figsize=(8, 8))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[11]:


test_loss, test_acc = resnet_model.evaluate(val_data, verbose=2)
print(test_acc)


# In[12]:


train_loss, train_acc = resnet_model.evaluate(train_data, verbose=2)
print(train_acc)


# In[ ]:




