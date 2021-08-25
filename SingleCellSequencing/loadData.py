
import os 

import cv2
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
#path2Images = '/Volumes/Seagate_Backup_Plus_Drive/PostDoc/Data'

def loadData(path2Images,w,h):
    Images = os.listdir(path2Images)
    imagesSet = np.ndarray(shape=(len(Images),w,h,3),dtype=float)
    imageNames = []
    for i, image in tqdm(enumerate(Images)):
        if image.endswith(".png"):
            img = cv2.imread(os.path.join(path2Images,image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = cv2.resize(img,dsize=(w,h),interpolation=cv2.INTER_CUBIC)
            imagesSet[i,:] = img
            imageNames.append(image)
   
    x = int(0.3 * len(Images))
 
    val = imagesSet[:x,:]
    train = imagesSet[x:,:]
    df = pd.DataFrame()
    df['Validation'] = imageNames[:x]
    df.to_csv('Validation.csv',index=False)
    return train,val
