import keras
import json
import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator

from liveness import create_model

model_pth = 'models/livenessv1.0.h5'
dataset_pth = r'D:\images\fakenessDataset\test'

height = 32
width = 32

def main():
    model = create_model()
    model.load_weights(model_pth)
    generator = ImageDataGenerator()
    dataloader = generator.flow_from_directory(
        dataset_pth,
        batch_size=32,
        target_size=(height,width),
        class_mode=None,
        shuffle=False
    )
    filename = dataloader.filenames
    y_pred = np.argmax(model.predict_generator(dataloader),axis=1).astype(np.float32)
    
    print(f'[INFO] 一共{len(filename)}个图片')
    print(f'[INFO] 一共{len(y_pred)}个结果')
    
    result = {}

    for i in range(len(y_pred)):
        result[filename[i]] = float(y_pred[i])
    
    print(result)

    with open('result/res2.json','w') as f:
        json.dump(result,f)

def display():

    with open('result/res2.json','r') as f:
        result = json.load(f)
    
    positives = []
    for k,v in result.items():
        if v == 0:
            positives.append(k)

    print(positives)

    for img in positives:
        pth = os.path.join(dataset_pth,img)
        print(pth)
        image = cv2.imread(pth,1)
        print(image.size)
        cv2.imshow('image',image)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break
        cv2.destroyAllWindows()
    
    print(positives)

if __name__ == '__main__':
    # main()
    display()

