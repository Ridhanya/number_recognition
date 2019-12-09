# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:42:42 2019

@author: RIDHANYA
"""
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
#import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
"""image_index = 5555 
print(y_train[image_index])
plt.imshow(x_train[image_index])
print(x_train.shape)
print(y_train.shape)"""
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#print(x_train.shape)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = (28, 28, 1)
#print('x_train shape:', x_train.shape)
#print('Number of images in x_train', x_train.shape[0])
#print('Number of images in x_test', x_test.shape[0])
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=3)
print(model.evaluate(x_test, y_test))



