import numpy as np
import os
import PIL
import h5py
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.python.keras.layers import Dense, Flatten
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import Adam
train_path = 'D:\MLA\MC_train'
test_path = 'D:\MLA\MC_valid'
img_height,img_width=180,180
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_path,image_size=(img_height, img_width),batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(test_path,image_size=(img_height, img_width),batch_size=batch_size)
class_names = train_ds.class_names
model = tf.keras.models.Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=5,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

model.add(pretrained_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='softmax'))
model.summary()
# resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs=4
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  steps_per_epoch=8
)
# import pickle
# pickle_out = open("model.pkl", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()
# print(class_names)
model.save('mla.h5')