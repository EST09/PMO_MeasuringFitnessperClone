import plotly
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
from os.path import expanduser
import pandas as pd

home = expanduser("~")
path22D = home + "/2D-SNU-668"

# cmap = plt.get_cmap('rainbow')

cmap = matplotlib.colors.ListedColormap(["maroon", "red", "darkorange", "yellow", "lime", "darkgreen", "navy", "cyan", "fuchsia", "indigo"])

cmap.set_under('white')


# for folder in os.listdir(path22D):
#     print(folder)
#     if os.path.isdir(os.path.join(path22D, folder)) == True: 
#         for filename in os.listdir(os.path.join(path22D, folder)):
#             if filename.endswith(".csv"):
#                 print(filename)
#                 image_name = filename.replace("csv", "png")
#                 df = pd.read_csv(os.path.join(path22D, folder, filename))
#                 array = np.zeros((400, 400))
#                 for x, y, z in zip(df["x"], df["y"], df[df.columns[3]]): 
#                     array[int(x), int(y)]=z
#                     if np.amax(array) > 10: 
#                         print("max more than 10", np.amax(array))
#                     if os.path.isfile(os.path.join(path22D, folder, image_name)) ==True:
#                         continue
#                     else:
#                         plt.imsave(os.path.join(path22D, folder, image_name), array, vmin=1, vmax=10, cmap=cmap)

# This is only printing out one pixel - why???? Fix later
            
df = pd.read_csv('ABC-family_proteins_mediated_transport.csv')

print(df.head())

# fig = px.scatter_3d(df, x='x', y='y', z='z',
#               color='ABC-family proteins mediated transport')
# fig.update_traces(marker={'size': 3})
# fig.show()

array = np.zeros((400, 400))
for x, y, z in zip(df["x"], df["y"], df[df.columns[3]]): 
    array[int(x), int(y)]=z

if np.amax(array) > 10: 
    print("max more than 10")

#cmap = plt.get_cmap('rainbow')
cmap.set_under('white')

#plt.imsave('test.png', array, vmin=1, vmax=10, cmap=cmap)

im = plt.imshow(array, vmin=1, vmax=10, cmap=cmap)
plt.colorbar(im, extend='min')
plt.show()
