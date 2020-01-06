# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:59:37 2020

@author: Pranav
"""
#Import modules
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras import utils 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras import optimizers

#Load and split MNIST data
(trainX, trainY), (testX, testY) = mnist.load_data()
print("TrainX shape {} TrainY shape {}".format(trainX.shape, trainY.shape))
print("TestX shape {} TestY shape {}".format(testX.shape, testY.shape))

#Show samples
plt.imshow(trainX[20]) #4
plt.imshow(trainX[21]) #0
plt.imshow(trainX[22]) #9

#Reshaping data
trainX = trainX.reshape(60000, 28, 28, 1)
testX = testX.reshape(10000, 28, 28, 1)

trainY = utils.to_categorical(trainY)
testY = utils.to_categorical(testY)

#Building the CNN Model
model = Sequential()
#Adding Convolutional Layer
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
#Adding Pooling Layer
model.add(MaxPooling2D((2, 2)))
#Adding Convolutional Layer
model.add(Conv2D(32, kernel_size=3, activation='relu'))
#Adding Pooling Layer
model.add(MaxPooling2D((2, 2)))
#Flattening and Dense-Fully Connected Layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#Compile model with Adam Optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics =['accuracy'])

#Train the model
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3)

#Accuracy Metrics
_, acc = model.evaluate(testX, testY)
print("Accuracy is {}%".format(acc*100))

#Predictions for the sample images
model.predict(trainX[20:21])
model.predict(trainX[21:22])
model.predict(trainX[22:23])
