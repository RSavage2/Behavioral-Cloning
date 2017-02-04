"""
Created on Sat Jan 28 15:26:31 2017

@author: R.Savage
"""
import numpy as np
import pandas as pd
import os
import json
from skimage.exposure import adjust_gamma
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy import ndimage
from scipy.misc import imresize
from keras.layers.advanced_activations import ELU


# Get steering angles as listed on the example in the behavioral course
holder = pd.read_csv('driving_log.csv',header=None)
holder.columns = ('Middle ','Left ','Right','Steering Angle','Throttle','Brake','Speed') 
holder = np.array(holder['Steering Angle'])

# Seperate data into left right, middle
imgs = np.asarray(os.listdir("IMG/"))
imgs = imgs[1:]
middle = np.ndarray(shape=(len(holder), 20, 64, 3))
right = np.ndarray(shape=(len(holder), 20, 64, 3))
left = np.ndarray(shape=(len(holder), 20, 64, 3))

# Images are resized to 32x64 
count = 0
for img in imgs:
    img_file = os.path.join('IMG', img)
    if img.startswith('center'):
        img_data = ndimage.imread(img_file).astype(np.float32)
        middle[count % len(holder)] = imresize(img_data, (32,64,3))[12:,:,:]
    elif img.startswith('right'):
        img_data = ndimage.imread(img_file).astype(np.float32)
        right[count % len(holder)] = imresize(img_data, (32,64,3))[12:,:,:]
    elif img.startswith('left'):
        img_data = ndimage.imread(img_file).astype(np.float32)
        left[count % len(holder)] = imresize(img_data, (32,64,3))[12:,:,:]
    count += 1

# combine training dataset with labels
X_train = np.concatenate((middle, right, left),axis= 0) 
y_train = np.concatenate((holder, (holder - .08), (holder + .08)), axis = 0) 
X_train = adjust_gamma(X_train)
mirror = np.ndarray(shape=(X_train.shape))
count = 0
for i in range(len(X_train)):
    mirror[count] = np.fliplr(X_train[i])
    count += 1
mirror.shape
mirror_angles = y_train * -1
X_train = np.concatenate((X_train, mirror), axis=0)
y_train = np.concatenate((y_train, mirror_angles),axis=0)

# test split taken from Traffic sign classifier 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.02) #0.05

# architecture      NVidia model served as a basis
model = Sequential()
model.add(BatchNormalization(axis=1, input_shape=(20,64,3)))
model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(1,2), activation='relu'))
model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='relu'))
model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(.5)) #.5 for training 1 for validation added to reduce overfitting
model.add(ELU())      #model.add(Activation('relu'))
model.add(Dense(10))
model.add(ELU())     #model.add(Activation('relu'))
model.add(Dense(1))
model.summary()
adam = Adam(lr=0.0001) # adam optimizer, learning rate of .0001 taken from examples in course work
model.compile(loss='mse',optimizer=adam)



checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# Train model for 15 epochs and a batch size of 128 taken from LeNet 5 architecture
#10 - 20* set at 10 for LeNet 5 needed more upped it by 5 
model.fit(X_train, y_train,nb_epoch=15, verbose=1,batch_size=128,shuffle=True,validation_data=(X_val, y_val),callbacks=[checkpoint, callback])

json_string = model.to_json()   # save as a json file
with open('model.json', 'w') as jsonfile:
    json.dump(json_string, jsonfile)
print("Model Saved")