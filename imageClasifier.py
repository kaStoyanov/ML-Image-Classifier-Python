import tensorflow as tf
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
# print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout


import cv2
import imghdr
data_dir = 'data'
print(os.listdir(data_dir))
image_exts = ['jpeg','jpg','bmp','png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path = os.path.join(data_dir,image_class,image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                os.remove(image_path)
        except Exception as e:
            print('issue found'.format(image_path))
            
            
            
# tf.data.Dataset
data = tf.keras.utils.image_dataset_from_directory('data')
# print(data)
data_iteratior = data.as_numpy_iterator()
# print(data_iteratior)
batch = data_iteratior.next()
# print(batch)
# scaled = batch[0] / 255
data = data.map(lambda x,y: (x/255,y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()
# Training models size
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
print(len(data))
print(train_size,test_size,val_size)
# Models

train = data.take(train_size)
val = data.take(train_size).take(val_size)
test = data.take(train_size+val_size).take(test_size)

model = Sequential()
# Input layer : Conv, with relu function 
model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())
# 2nd
model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D())
# 3rd
model.add(Conv2D(16,(3,3),1,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
print(model.summary())

logdir='logs'
# Training the model epochs 20 
tensoreboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train,epochs=20,validation_data = val,callbacks=[tensoreboard_callback])

# fig = plt.figure()
# plt.plot(hist.history['loss'],color='teal',label='loss')
# plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
# fig.suptitle('Loss',fontsize=20)
# plt.legend(loc='upper left')
# plt.show()