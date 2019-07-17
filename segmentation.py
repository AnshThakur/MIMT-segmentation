

import keras
from keras import backend as K
import keras.layers
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda, RepeatVector
from keras.layers.convolutional import Conv2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import merge, Input, GRU, TimeDistributed, GlobalAveragePooling2D,Concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers.merge import multiply
from keras.layers import concatenate
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Activation, GRU, TimeDistributed, Bidirectional
initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2)
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
from keras import regularizers
from keras import optimizers
############

import numpy as np

###########

# Loss function


# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out



def train_model_1():  
    inp=Input(shape=(40,438,1), name='in_layer')
    a2=Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(inp)
    a2=BatchNormalization(axis=-1)(a2)

    a1=Conv2D(32, (3, 3), strides=(2, 1), activation='relu', kernel_initializer=initializer,padding='same')(a2)
  
    ## 2nd block
    
    a1=Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)
    a1=BatchNormalization(axis=-1)(a1)
 
    a1=Conv2D(32, (3, 3), strides=(2, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)

    ## 3rd block

    a1=Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)
    a1=BatchNormalization(axis=-1)(a1)

    a1=Conv2D(32, (3, 3), strides=(2, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)
    
    #cnn=Dropout(0.25)(cnn)
    a1=Conv2D(32,(3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)
    a1=BatchNormalization(axis=-1)(a1)
  
    a1=Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)

    a1=Conv2D(32,(3, 3), strides=(5, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)
    ## fourth block
    a1 = Reshape((438,32))(a1) 
   
    # Attention 
    #Dum=Conv2D(32, (40, 1), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(inp)
    #Dum=BatchNormalization(axis=-1)(Dum)
    #Dum=Conv2D(32, (8, 1), strides=(8, 1), activation='relu', kernel_initializer=initializer,padding='same')(Dum)
    #Dum = Reshape((438, 32))(Dum) 
    '''
    att = TimeDistributed(Dense(64, activation='sigmoid'))(a1)
    att = TimeDistributed(Dense(32, activation='sigmoid'))(att)
    att = TimeDistributed(Dense(1, activation='sigmoid',name='localization_layer'))(att)
    att=Flatten()(att)
    att= RepeatVector(32)(att)
    att = Reshape((438,32))(att) 
    ############## apply attention 
    merge1 = multiply([a1,att])
    '''
    ######## instance
    
    dnn = TimeDistributed(Dense(128, activation='sigmoid'))(a1)
    dnn = TimeDistributed(Dense(64, activation='sigmoid'))(dnn)
    cla1 = TimeDistributed(Dense(2))(dnn)
    cla=Activation('sigmoid')(cla1)
    att=Activation('softmax',name='localization_layer')(cla1)
    out= Lambda(outfunc, output_shape=(2,))([cla, att])
    out=Activation('softmax')(out)
    #out = AutoPool1D(axis=1,kernel_constraint=keras.constraints.non_neg())(cla1)
    

    ######## bag
 
    feature=Conv1D(32, (438), strides=(1), activation='relu')(a1)
    feature=Reshape((1, 1,32))(feature)
    #feature=Dropout(0.50)(feature)
    feature=Conv2D(32,(1,1),activation = 'sigmoid')(feature)
    #feature=Dropout(0.50)(feature)
    feature=Conv2D(2,(1,1),activation = 'softmax')(feature)
    out1=Reshape((2,))(feature)

    singleOut = Concatenate(axis=1)([out,out1])
    model = Model(inp,singleOut)
    model.summary()
    return model




def run_func(func, x, batch_size):
    pred_all = []
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    for i1 in range(batch_num):
        batch_x = x[batch_size * i1 : batch_size * (i1 + 1)]
        [preds] = func([batch_x, 0.])
        pred_all.append(preds)
    pred_all = np.concatenate(pred_all, axis=0)
    return pred_all




model=train_model_1() 
model.load_weights('./birdvox/models/surrogate.h5')

model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy'])

segs=[]

'''
bird_data = np.load('./birdvox/dbad_fcall_feature_40_20.npy')
bird_label = np.load('./birdvox/dbad_fcall_label_40_20.npy')

nbird_data = np.load('./birdvox/dbad_nfcall_feature_40_20.npy')
nbird_label = np.load('./birdvox/dbad_nfcall_label_40_20.npy')


feature_wab = np.concatenate((bird_data,nbird_data),axis=0)
'''
x_train=sio.loadmat('train_data.mat');
feature_wab=x_train['x_train']
print(feature_wab.shape)

in_layer = model.get_layer('in_layer')
print(in_layer)
loc_layer = model.get_layer('localization_layer')
print(loc_layer)
func = K.function([in_layer.input, K.learning_phase()],[loc_layer.output])
pred3d = run_func(func, feature_wab, batch_size=20)
print(pred3d.shape)
segs.append(pred3d)
sio.savemat('./wab_segments.mat',{'segs':pred3d})








