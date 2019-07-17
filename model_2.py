

import keras
from keras import backend as K
import keras.layers

from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda, RepeatVector
from keras.layers.convolutional import Conv2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import merge, Input, GRU, TimeDistributed, GlobalAveragePooling2D, Concatenate
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
from data_generator import RatioDataGenerator
import tensorflow as tf
from autopool import AutoPool1D

from keras.losses import mean_squared_error, binary_crossentropy

from keras.metrics import categorical_accuracy

############

import numpy as np
from sklearn.utils import shuffle
###########


# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out

###########################










def train_model_1():  
    inp=Input(shape=(40,500,1), name='in_layer')
    ## feature extraction strand
    a2=Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(inp)
    a2=BatchNormalization(axis=-1)(a2)

    a1=Conv2D(32, (3, 3), strides=(2, 1), activation='relu', kernel_initializer=initializer,padding='same')(a2)
    
    
    
    a1=Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)
    a1=BatchNormalization(axis=-1)(a1)
 
    a1=Conv2D(32, (3, 3), strides=(2, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)


    a1=Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)
    a1=BatchNormalization(axis=-1)(a1)

    a1=Conv2D(32, (3, 3), strides=(2, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)
    
   
    a1=Conv2D(32,(3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)
    a1=BatchNormalization(axis=-1)(a1)
  
    a1=Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)

    a1=Conv2D(32,(3, 3), strides=(5, 1), activation='relu', kernel_initializer=initializer,padding='same')(a1)
    
    a1 = Reshape((500,32))(a1) 
   

    ######## instance
    
    dnn = TimeDistributed(Dense(128, activation='sigmoid'))(a1)
    dnn = TimeDistributed(Dense(64, activation='sigmoid'))(dnn)
    cla1 = TimeDistributed(Dense(2))(dnn)
    cla=Activation('sigmoid')(cla1)
    att=Activation('softmax',name='segmentation_layer')(cla1)
    out= Lambda(outfunc, output_shape=(2,))([cla, att])
    out=Activation('softmax')(out)
    #out = AutoPool1D(axis=1,kernel_constraint=keras.constraints.non_neg())(cla1)
    

    ######## bag
 
    feature=Conv1D(32, (500), strides=(1), activation='relu')(a1)
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


def metric(y_true, y_pred):
    out = y_pred[:, :2]
    return categorical_accuracy(y_true,out)



def weightedLoss(yTrue,yPred):   
    out = yPred[:, :2]
    out1 = yPred[:, 2:4] 
    mse=mean_squared_error(out1, out)
    crossentropy = binary_crossentropy(yTrue, out1)
    crossentropy1 = binary_crossentropy(yTrue, out)
    return K.cast(0.5,'float32')*crossentropy +  K.cast(0.5,'float32')*mse #+K.cast(0.1,'float32')*crossentropy1






model=train_model_1()
adam = optimizers.Adam(lr = 1e-4)

model.compile(loss=weightedLoss, optimizer=adam, metrics=[metric])
#model.load_weights('birdvox_weights.h5')

####


# load nips data
bird_data = np.load('./birdvox/dbad_fcall_feature_40_20.npy')
bird_label = np.load('./birdvox/dbad_fcall_label_40_20.npy')

nbird_data = np.load('./birdvox/dbad_nfcall_feature_40_20.npy')
nbird_label = np.load('./birdvox/dbad_nfcall_label_40_20.npy')


feature_train = np.concatenate((bird_data,nbird_data),axis=0)
label_train = np.concatenate((bird_label,nbird_label),axis=0)

feature_train,label_train = shuffle(feature_train,label_train)
feature_train,label_train = shuffle(feature_train,label_train)
feature_train,label_train = shuffle(feature_train,label_train)
feature_train,label_train = shuffle(feature_train,label_train)

label_train = to_categorical(label_train, 2)




x_train, x_test, y_train, y_test = train_test_split(feature_train, label_train, test_size=0.2, shuffle=True) # assign validation if needed
print(x_train.shape)
print(x_test.shape)




filepath="./MTML_model.h5"


checkpoint = ModelCheckpoint(filepath,verbose=1, monitor='val_loss', save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(x_train, y_train,callbacks=callbacks_list,validation_data=(x_test,y_test), epochs=100, batch_size=32)

train_loss=history.history['loss']
val_loss=history.history['val_loss']

sio.savemat('./plots/surrogate/su_loss.mat',{'train_loss':train_loss})

sio.savemat('./plots/surrogate/val_su_loss.mat',{'val_loss':val_loss})












from sklearn.metrics import classification_report
import numpy as np

Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
pred1 = model.predict(x_test)
print(pred1.shape)

## prediction from weakly supervised strand
pred = np.argmax(pred1[:, :2], axis=1)
print(Y_test.shape)
print(pred.shape)
print(classification_report(Y_test,pred))


## prediction from supervised strand
print('-----------------------------------------')
pred = np.argmax(pred1[:, 2:4], axis=1)
print(Y_test.shape)
print(pred.shape)
print(classification_report(Y_test,pred))
############################ plot con

