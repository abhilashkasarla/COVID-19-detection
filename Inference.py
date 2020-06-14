#!/usr/bin/env python
# coding: utf-8

# In[2]:


from chexnet import get_chexnet_model
from keras.layers import Input, Dense, Dropout
from keras.utils import print_summary
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from generator import AugmentedImageSequence
from test_CheXNet import target_classes
#from weights import get_class_weights


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image


# In[4]:


from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[5]:


from chexnet import get_chexnet_model


# In[6]:


def get_3class_model():
    # get base model, model
    base_model, chexnet_model = get_chexnet_model()

    x = base_model.output
    # Dropout layer
    #x = Dropout(0.2)(x)
    # one more layer (relu)
#     x = Dense(512, activation='relu')(x)

    predictions = Dense(
        3,
        activation="sigmoid")(x)

    # this is the model we will use
    model = Model(
        inputs=base_model.input,
        outputs=predictions,
    )

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all base_model layers
    for layer in base_model.layers:
        layer.trainable = False

    # initiate an Adam optimizer
    opt = Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False
    )

    # Let's train the model using Adam
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    return base_model, model


def run_inference():
    # In[7]:


    _, best_model = get_3class_model()


    # In[8]:


    best_model.load_weights("models/3class_best.hdf5")


    # In[9]:


    TRAIN_PATH = "../chest_xray/train"
    VAL_PATH = "../chest_xray/test"


    # In[15]:


    # os.listdir(VAL_PATH + "/Covid/")


    # In[18]:


    class_ids = {0:'COVID-19', 1:'Normal', 2:'Pneumonia'}


    # In[25]:


    img_path = VAL_PATH + '/Covid/16654_4_1.jpg'
    img_path


    # In[17]:


    img=image.load_img(img_path,target_size=(224,224))
    img=image.img_to_array(img)/255.0
    img=np.expand_dims(img,axis=0)


    # In[20]:


    pred=best_model.predict(img)
    idx = np.argmax(pred[0])


    # In[24]:


    print(class_ids[idx] + ' with ' + str(pred[0][idx]) + ' probability')

if __name__ == '__main__':
    run_inference()