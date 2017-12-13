# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:08:52 2017

@author: Diwas.Tiwari
"""

import keras
from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
## Checking the basic info of the images ##
print('Shape of the Training data:', train_images.shape, train_labels.shape)
print('Shape of the Test data:', test_images.shape, test_labels.shape)

import numpy as np
class_in_data = np.unique(train_labels)
number_classes = len(class_in_data)
print('No of classes of image are: ', class_in_data)
print('length of the classes: ', number_classes)

import matplotlib.pyplot as plt
from matplotlib import pyplot
plt.figure(figsize = [7,7])
plt.subplot(121)
plt.imshow(train_images[5,:,:], cmap = 'gray')
plt.title('Ground truth: {}'.format(train_labels[5]))
plt.subplot(122)
plt.imshow(test_images[5,:,:], cmap = 'gray')
plt.title('Ground truth: {}'.format(test_labels[5]))
plt.show()
## Now preparing data for the Neural network ##
arr_dim = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], arr_dim)
test_data = test_images.reshape(test_images.shape[0], arr_dim)
## Data Type Conversion ##
train_data = train_data.astype('float32') ## Type conversion for ease of classification ##
test_data = test_data.astype('float32')

train_data = train_data/255 ## Scaling pixel intensities betwenn 0 to 1. ##
test_data = test_data/255
 ## One Hot encoding the Categorical Values for Multi-Class Classification ##
 
 from keras.utils import to_categorical
 
 train_label_one_hot_encoded = to_categorical(train_labels)
 test_label_one_hot_encoded = to_categorical(test_labels)
 
 ## Building of 2- Layer Deep neural Network ##
 
 from keras.models import Sequential
 from keras.layers import Dense
 model = Sequential()
 model.add(Dense(512,activation = 'relu', input_shape =(arr_dim,)))
 model.add(Dense(256,activation = 'relu'))
 model.add(Dense(number_classes,activation = 'softmax'))
 model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
 ## Training the network ##
 
 history = model.fit(train_data, train_label_one_hot_encoded, batch_size = 10, epochs = 10,
                       validation_data = (test_data, test_label_one_hot_encoded))
 [loss,accuracy] = model.evaluate(test_data, test_label_one_hot_encoded)
 print('Evaluation Results are: {}'.format(loss,accuracy))
 
 ## Ploting training and validation accuracy ##
print(history.history.keys())
plt.figure(figsize = [8,8])
plt.plot(history.history['acc'], color = 'magenta')
plt.plot(history.history['val_acc'], color = 'cyan')
plt.title('Accuracy Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training_Acc', 'Validation_Acc'])
plt.show()   
## Plotting training and validation loss ##
plt.figure(figsize = [8,8])
plt.plot(history.history['loss'], color = 'green')
plt.plot(history.history['val_loss'], color = 'red')
plt.title('Loss_Graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Training_loss','Validation Loss'])
plt.show()

#### MNIST Text Classification Done ####

 
 
 
 
 
 