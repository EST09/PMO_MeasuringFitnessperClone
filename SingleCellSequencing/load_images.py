import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Input, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg

import os
from os.path import expanduser
from tqdm import tqdm
import cv2


home = expanduser("~")
img_folder = home + "/test_images"

def loadData(path2Images,w,h):
    Images = os.listdir(path2Images)
    print(len(Images))
    imagesSet = np.ndarray(shape=(len(Images),w,h,3),dtype=float)
    imageNames = []
    for i,image in tqdm(enumerate(Images)):
        img = cv2.imread(os.path.join(path2Images,image))
        img = cv2.resize(img,dsize=(w,h),interpolation=cv2.INTER_CUBIC) #https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
        imagesSet[i,:] = img
        imageNames.append(image)
    
    plt.imshow(imagesSet[1,:])
    plt.show()

loadData(img_folder, 400, 400)

    # imageNames = []
    # for i,image in tqdm(enumerate(Images)):
    #     img = cv2.imread(os.path.join(path2Images,image))
    #     img = cv2.resize(img,dsize=(w,h),interpolation=cv2.INTER_CUBIC)
    #     imagesSet[i,:] = img
    #     imageNames.append(image)
 
    # #imagesSet = np.moveaxis(imagesSet,3,1)
   
    # x = int(0.3 * len(Images))
 
    # val = imagesSet[:x,:]
    # train = imagesSet[x:,:]
    # df = pd.DataFrame()
    # df['Validation'] = imageNames[:x]
    # df.to_csv('Validation.csv',index=False)
    # return train,val


