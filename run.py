#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Rockstar He
#Date: 2020-07-03
#Description:
import tensorflow as tf
keras = tf.keras
import keras
import json
import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator

from liveness import create_model

class Fakeness():
    """
    翻拍文件侦测模型1.0
    """
    def __init__(self,model_path = r'models\livenessv1.0.h5'):
        self.model = create_model()
        self.model.load_weights(model_path)
    
    def predict(self,image):
        """
        单张预测图片

        param image(str):图片路径
        return res(float):图片得分，0为翻拍，1为正常
        """
        img = cv2.imread(image,1)
        img = cv2.resize(img,(32,32))
        img = np.expand_dims(img,axis = 0)
        res = np.argmax(self.model.predict(img),axis=1).astype(np.float32)
        print(f'[INFO] 这张照片得分为{res[0]}!')
        
        return res

    def predict_on_batch(self,image_folder_path):
        """
        批量预测图片,并保存结果至result.json

        param image_folder_path(str):图片文件夹路径，请注意应保证此路径下仍为文件夹，不能直接是图片
        return result(dict):字典，键为图片路径，值为图片得分，0为翻拍，1为正常
        """
        generator = ImageDataGenerator()
        dataloader = generator.flow_from_directory(image_folder_path,batch_size=32,target_size=(32,32),class_mode=None,shuffle=False)
        filename = dataloader.filenames
        print(f'[INFO] 一共找到{len(filename)}张图片')
        y_pred = np.argmax(self.model.predict_generator(dataloader),axis=1).astype(np.float32)
        print(f'[INFO] 预测结束！')
        
        result = {}
        for i in range(len(y_pred)):
            result[os.path.join(image_folder_path,filename[i])] = float(y_pred[i])
        
        with open('result.json','w') as f:
            json.dump(result,f)

        return result

if __name__ == "__main__":
    fakeness = Fakeness()
    # fakeness.predict(r'C:\Users\192.168.2.52\fakeness\testimage\test\00a7ee6b-8e49-4157-966e-540edad3a5df.jpeg')
    fakeness.predict_on_batch(r'D:\images\fakenessDataset\test')
