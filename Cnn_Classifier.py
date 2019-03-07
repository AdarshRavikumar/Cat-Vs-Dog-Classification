# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier=Sequential()
# Adding Layer 1
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding Layer 2
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding Layer 3

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flattening

classifier.add(Flatten())

# Full Connection

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))


# Compile
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Pre Processing

from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage
#Augmentation 
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

#Training and Test Set



training_set=train_datagen.flow_from_directory('/home/adarsh/Downloads/CNN/dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')
test_set=test_datagen.flow_from_directory('/home/adarsh/Downloads/CNN/dataset/test_set',target_size=(64,64),batch_size=32,class_mode='binary')

# fit the model
classifier.fit_generator(training_set,samples_per_epoch=8000,nb_epoch=30,validation_data=test_set,nb_val_samples=2000)
