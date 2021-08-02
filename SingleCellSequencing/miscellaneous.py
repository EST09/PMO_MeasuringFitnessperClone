import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import expanduser
import plotly
import plotly.express as px
import PIL
import cv2


#understanding colours in my image set

# home = expanduser("~")
# img_path = "smaller_test_imgs/Phase_I_-_Functionalization_of_compounds.png"

# img = cv2.imread(os.path.join(home, img_path))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img.shape)
# plt.imshow(img)
# plt.show()




#resizing images and creating smaller_image folder

home = expanduser("~")
path = home + "/Andor_Rotation/PMO_MeasuringFitnessperClone/data/3Dbrightfield/allencell/D03_FijiOutput"

df = pd.read_csv(os.path.join(path, 'mito_anothercell.csv'))

print(df.head())
print(df['CX (pix)'].min(), df['CX (pix)'].max())
print(df['CY (pix)'].min(), df['CY (pix)'].max())

path2clone = home + "/2D-SNU-668/Clone_0.0024504_ID107171"
path2smaller = home + "/smaller_test_imgs"

cmap = matplotlib.colors.ListedColormap(["maroon", "red", "darkorange", "yellow", "lime", "darkgreen", "navy", "cyan", "fuchsia", "indigo"])
cmap.set_under('white')

f = np.zeros((3, 3))
print(f)

for file in os.listdir(path2clone):
    if file.endswith(".csv"):
        
        image_name = file.replace("csv", "png")
        
        df = pd.read_csv(os.path.join(path2clone, file))
        print(df.head())
        array = np.zeros((400, 400))
        print(array.shape)
        print(array)
        for x, y, z in zip(df["x"], df["y"], df[df.columns[3]]): 
            array[int(x), int(y)]=z
            sn_array = array[50:350, 150:400]

            bulk = np.zeros((300, 50))
            n_array = np.append(sn_array, bulk, axis=1)

            if np.amax(n_array) > 10: 
                print("max more than 10", np.amax(n_array))
            
            if os.path.isfile(os.path.join(path2smaller)) ==True:
                continue
            else:
                plt.imsave(os.path.join(path2smaller, image_name), n_array, vmin=1, vmax=10, cmap=cmap)
                



# median = df['CZ (pix)'].median()
# subset = df[(df["CZ (pix)"] == median)]

# fig = px.scatter_3d(df, x='CX (pix)', y='CY (pix)', z='CZ (pix)')
# fig.update_traces(marker={'size': 3})
# fig.show()