#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from scipy import signal
import matplotlib.colors as colors
from sklearn.utils import shuffle
import mne
import gc
from glob import glob


# # Pre-processing

# In[2]:


all_raw_file = glob('dataverse_files/*.edf')


# In[3]:


len(all_raw_file)


# In[4]:


all_raw_file[1]


# In[5]:


healthy_raw_file = [i for i in all_raw_file if 'h' in i.split('\\')[1]]
sc_raw_file = [i for i in all_raw_file if 's' in i.split('\\')[1]]


# In[6]:


print(len(healthy_raw_file))
print(len(sc_raw_file)) 


# In[7]:


ch_names = ['Fp2', 'F8', 'Fp1', 'F7', 'F4', 'F3', 'Fz']
ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg']


# In[8]:


info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=ch_types)


# In[9]:


montage_kind = "standard_1020"


# In[10]:


montage =  mne.channels.make_standard_montage(montage_kind)


# In[11]:


def preprocess_data(filepath):
    data = mne.io.read_raw_edf(filepath, preload= True)
    data.filter(l_freq= 0.5, h_freq= 4.5)
    
    data.info = info
    data.set_montage(montage, match_case=False)
    
    epoch = mne.make_fixed_length_epochs(data, duration=5, preload="False")
    return epoch


# In[12]:


get_ipython().run_cell_magic('capture', '', 'healthy_epochs = [preprocess_data(i) for i in healthy_raw_file]\nsc_epochs = [preprocess_data(i) for i in sc_raw_file]')


# # Time-frequency transformation

# In[13]:


total_img = 0
def tf_transform(epoch):
    global total_img
    stft_data = []
    for e in epoch.get_data():
        flatten_epoch = e.flatten()
        stft_data += [signal.stft(flatten_epoch, 250, window='hann', nperseg=40 )]
        total_img += 1
    return stft_data


# In[14]:


get_ipython().run_cell_magic('capture', '', 'healthy_tf = [tf_transform(i) for i in healthy_epochs]')


# In[15]:


print('Healthy data amount is', total_img)


# In[16]:


get_ipython().run_cell_magic('capture', '', 'sc_tf = [tf_transform(i) for i in sc_epochs]')


# In[17]:


print('Total amount is', total_img)


# # Create dataset

# In[18]:


training_count = round(total_img*80/100)
test_count = round(total_img*10/100) +training_count


# In[19]:


print(training_count)
print(test_count)


# In[20]:


print(len(healthy_tf))
print(len(sc_tf))


# In[21]:


healthy_tf_flatten = sum(healthy_tf, [])
sc_tf_flatten = sum(sc_tf, [])
print(len(healthy_tf_flatten))
print(len(sc_tf_flatten))


# In[22]:


healthy_label = [0 for i in range(len(healthy_tf_flatten))]
sc_label = [1 for i in range(len(sc_tf_flatten))]
print(len(healthy_label))
print(len(sc_label))


# In[23]:


data_list = healthy_tf_flatten+sc_tf_flatten
label_list = healthy_label+sc_label


# In[24]:


shffled_data_list, shffled_label_list = shuffle(data_list, label_list, random_state=0)


# In[25]:


print(len(shffled_data_list))
print(len(shffled_label_list))


# In[26]:


no_round = 0

training_sc = []
training_healthy = []
val_sc = []
val_healthy = []
test_sc = []
test_healthy = []

def categorize_data(tfs, labels):
    global no_round
    global training_sc
    global training_healthy
    global val_sc
    global val_healthy
    global test_sc
    global test_healthy
   
    for tf, label in zip(tfs, labels):
            no_round += 1  
            if(no_round <= training_count):
                    if(label==0):
                        training_healthy.append(tf)
                    else:
                        training_sc.append(tf)

            if(no_round <= test_count and no_round > training_count):
                    if(label==0):
                        test_healthy.append(tf)
                    else:
                        test_sc.append(tf)

            if(no_round <= total_img and no_round > test_count):
                    if(label==0):
                        val_healthy.append(tf)
                    else:
                        val_sc.append(tf)


# In[27]:


categorize_data(shffled_data_list,shffled_label_list)


# In[28]:


print(len(training_sc))
print(len(training_healthy))

print(len(test_sc))
print(len(test_healthy))

print(len(val_sc))
print(len(val_healthy))


# In[29]:


no_round = 0
no_training = 0
no_testing = 0
no_validation = 0
def tf_plot_save(tfs,ty, label):
    
    no_round = 0
    for tf in tfs:
            no_round += 1  
            fig = plt.figure(figsize=(9,5))
            spec = plt.pcolormesh(tf[1], tf[0], np.abs(tf[2]), 
                              norm=colors.PowerNorm(gamma=1./8.),
                              cmap=plt.get_cmap('magma'))

            plt.xticks([])
            plt.yticks([]) 
            
            if(ty=='training'):
                if(label=='healthy'):
                    plt.savefig('dataset_frontal/training/healthy/h'+str(no_round)+'.png', pad_inches=-0.1, bbox_inches = 'tight')
                    plt.close()
                else:
                    plt.savefig('dataset_frontal/training/sc/sc'+str(no_round)+'.png',pad_inches=-0.1, bbox_inches = 'tight')
                    plt.close()
            elif(ty=='testing'):
                if(label=='healthy'):
                    plt.savefig('dataset_frontal/testing/healthy/h'+str(no_round)+'.png',pad_inches=-0.1, bbox_inches = 'tight')
                    plt.close()
                else:
                    plt.savefig('dataset_frontal/testing/sc/sc'+str(no_round)+'.png',pad_inches=-0.1, bbox_inches = 'tight')
                    plt.close()

            elif(ty=='val'):
                if(label=='healthy'):
                    plt.savefig('dataset_frontal/validation/healthy/h'+str(no_round)+'.png',pad_inches=-0.1, bbox_inches = 'tight')
                    plt.close()
                else:
                    plt.savefig('dataset_frontal/validation/sc/sc'+str(no_round)+'.png',pad_inches=-0.1, bbox_inches = 'tight')
                    plt.close()


# In[30]:


tf_plot_save(training_sc,'training','sc')


# In[31]:


tf_plot_save(training_healthy,'training','healthy')


# In[32]:


tf_plot_save(test_sc,'testing','sc')


# In[33]:


tf_plot_save(test_healthy,'testing','healthy')


# In[34]:


tf_plot_save(val_sc,'val','sc')


# In[35]:


tf_plot_save(val_healthy,'val','healthy')


# In[ ]:




