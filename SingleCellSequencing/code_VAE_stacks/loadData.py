
import os 

import cv2
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
from PIL import Image 
from PIL import GifImagePlugin 
from PIL import ImageSequence 

#path2Images = '/Volumes/Seagate_Backup_Plus_Drive/PostDoc/Data'

def loadData(path2Images,w,h):
    Images = os.listdir(path2Images)
    imagesSet = np.ndarray(shape=(len(Images),w,h,3),dtype=float)
    imageNames = []
    for i,image in tqdm(enumerate(Images)):
        img = cv2.imread(os.path.join(path2Images,image))
        img = cv2.resize(img,dsize=(w,h),interpolation=cv2.INTER_CUBIC)
        imagesSet[i,:] = img
        imageNames.append(image)
 
    #imagesSet = np.moveaxis(imagesSet,3,1)
   
    x = int(0.3 * len(Images))
 
    val = imagesSet[:x,:]
    train = imagesSet[x:,:]
    df = pd.DataFrame()
    df['Validation'] = imageNames[:x]
    df.to_csv('Validation.csv',index=False)
    return train,val
def loadDataGIF(path2Images,w,h):
    Images = os.listdir(path2Images)
    imagesSet = np.ndarray(shape=(len(Images),w,h,3,39),dtype=float)
    imageNames = []
    for i,image in tqdm(enumerate(Images)):
        ImageObject = Image.open(os.path.join(path2Images,image))
        frames =[frame.copy() for frame in ImageSequence.Iterator(ImageObject)]
        
        for frame in range(0,len(frames)):
            if frame > 38:
                continue
            else:
                img = frames[frame]
                img = img.convert(mode='RGB')
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,dsize=(w,h),interpolation=cv2.INTER_CUBIC)
                imagesSet[i,:,:,:,frame] = img 
                imageNames.append(image)

    x = int(0.3*len(Images)) 

    val = imagesSet[:x,:]
    train = imagesSet[x:,:]
    df = pd.read_csv('Validation.csv')
    df['ValidationGif'] = imageNames[:x]
    df.to_csv('Validation_updates.csv',index=False)
    return train, val

        
