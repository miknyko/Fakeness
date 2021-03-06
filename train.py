#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Rockstar He
#Date: 2020-07-03
#Description:
import tensorflow as tf
keras = tf.keras
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping ,ModelCheckpoint

from liveness import create_model

height = 32
width = 32
depth = 3

trainset = r'D:\images\fakenessDataset\train'
valset = r'D:\images\fakenessDataset\test'
ckp_path = r'models\ckp.h5'

def train():

    generator = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True
    )

    traindataloader = generator.flow_from_directory(
        trainset,
        batch_size=32,
        target_size=(height,width)
    )

    valdataloader = generator.flow_from_directory(
        valset,
        batch_size=32,
        target_size=(height,width)
    )

    train_ckp = ModelCheckpoint(ckp_path,monitor='val_acc')
    model = create_model()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print('=' * 40 + '开始训练' + '=' * 40)
    model.fit_generator(
        traindataloader,
        epochs=100,
        verbose=1,
        callbacks=[train_ckp],
        validation_data=valdataloader,
        workers=2

    )
    model.save_weights(r'models\liveness3.0.h5')



if __name__ == "__main__":
    train()



