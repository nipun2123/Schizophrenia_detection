#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout,AveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from keras.applications.imagenet_utils import decode_predictions 
from sklearn import preprocessing
from sklearn import metrics
import glob
import cv2
import os
import gc
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the dataset

# In[2]:


train_data = []
train_labels = [] 


# In[3]:


#load training data
for directory_path in glob.glob("dataset/training/*"):
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


#convert into numby array
train_data = np.array(train_data)
train_labels = np.array(train_labels)


# In[6]:


validation_data = []
validation_labels = [] 


# In[7]:


#load validation data
for directory_path in glob.glob("dataset/validation/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (508, 274))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        validation_data.append(img)
        validation_labels.append(label)


# In[8]:


#convert into numby array
validation_data = np.array(validation_data)
validation_labels = np.array(validation_labels)


# # Transform the labels

# In[9]:


#create a label encoder
le = preprocessing.LabelEncoder()


# In[10]:


#transform validation label into 0 or 1
le.fit(validation_labels)
y_test = le.transform(validation_labels)


# In[11]:


#transform training label into 0 or 1
le.fit(train_labels)
y_train = le.transform(train_labels)


# In[12]:


train_data.shape


# In[13]:


#reshape the training data array
train_data = train_data.reshape(train_data.shape[0],train_data.shape[2],train_data.shape[1],train_data.shape[3])


# In[14]:


#reshape the validation data array
validation_data = validation_data.reshape(validation_data.shape[0],validation_data.shape[2],validation_data.shape[1],validation_data.shape[3])


# # Feature extraction

# In[15]:


#create RestNet50 model
resnet_model = tf.keras.applications.ResNet50(include_top=False,

                   input_shape=(508, 274,3),

                   pooling='min',

                   weights='imagenet')

for each_layer in resnet_model.layers:

        each_layer.trainable=False
        


# In[16]:


print(resnet_model.summary())


# In[17]:


#extract the training features
X_train = resnet_model.predict(train_data)


# In[18]:


#extract the validation features
X_test = resnet_model.predict(validation_data)


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


#reshape training extracted feature array
X_train = np.reshape(X_train, (4617, 16*9*2048))


# In[22]:


#reshape validation extracted feature array
X_test = np.reshape(X_test, (577, 16*9*2048))


# # Classification

# In[23]:


#create SVM model
svm_model = svm.SVC(C=1, gamma=1, kernel='linear')


# In[24]:


#fit training data into SVM model
svm_model.fit(X_train,y_train)


# # Model evaluation

# In[25]:


#predit validation data
test_prediction = svm_model.predict(X_test)


# In[26]:


#get accuracy score for validation data
print("Accuracy:",metrics.accuracy_score(y_test, test_prediction)*100)


# In[27]:


#get precision and recall
print("Precision:",metrics.precision_score(y_test, test_prediction))

print("Recall:",metrics.recall_score(y_test, test_prediction))


# In[28]:


#get classification report
print(classification_report(y_test, test_prediction))


# In[29]:


#get confution matrix
cm = confusion_matrix(y_test, test_prediction, labels = svm_model.classes_)


# In[31]:


#plot confution matrix
cmd = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = svm_model.classes_)  
cmd.plot()
plt.show()

