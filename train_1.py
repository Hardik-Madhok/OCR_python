# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 00:30:39 2021

@author: Hardik
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import tensorflow as tf
from utils import read_images, extract_labels
'''In this code model is trained on the clipped symbols and their labels which were written in the previous code'''
images, names = read_images('new_dataset_3',128,100)
labels = extract_labels(names)
    
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.reshape(-1, 128, 100, 3)
x_test = x_test.reshape(-1, 128, 100, 3)

le = preprocessing.LabelEncoder()
le.fit(y_train)
'''encoder weights are saved for testing purpose'''
np.save('classes.npy', le.classes_)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)
y_train_enc = to_categorical(y_train_enc, num_classes=15)
y_test_enc = to_categorical(y_test_enc, num_classes=15)

model = Sequential([
    Conv2D(32,3, activation='relu', input_shape=(128,100,3)),
    # BatchNormalization(),
    # MaxPooling2D(pool_size=(3, 3)),
    
    # Conv2D(64,3, activation='relu'),
    # BatchNormalization(),
    # MaxPooling2D(pool_size=(3, 3)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dense(15,activation='softmax')  # activation change
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # Loss
              metrics=['accuracy'])

history = model.fit(x_train,y_train_enc,epochs = 20 , validation_data = (x_test, y_test_enc))
'''trained model is saved'''
model.save("model.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(20)
'''plots of training and validation accuarcy along wiht losses are made'''
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

    
