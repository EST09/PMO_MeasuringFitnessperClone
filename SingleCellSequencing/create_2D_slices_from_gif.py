import os
from os.path import expanduser
import pandas as pd

home = expanduser("~")
path2Identities = home + "/SNU-668"
path22D = home + "/2D-SNU-668"

for folder in os.listdir(path2Identities):
    print(folder)
    if not os.path.exists(os.path.join(path22D, folder)):
        os.mkdir(os.path.join(path22D, folder))
    if os.path.isdir(os.path.join(path2Identities, folder)) == True: 
        for filename in os.listdir(os.path.join(path2Identities, folder)):
            if filename.endswith(".txt"):
                new_name = filename.replace("txt", "csv")
                print(new_name)
                file = pd.read_csv(os.path.join(path2Identities, folder, filename),delimiter="\t")
                median = file['z'].median()
                subset = file[(file["z"] == median)]
                subset.to_csv(os.path.join(path22D, folder, new_name), index=False)
                
                

# import plotly
# import plotly.express as px
# import pandas as pd
# import numpy as np

# df = pd.read_csv('ABC-family_proteins_mediated_transport.txt', delimiter = "\t")

# print(df.head())

# fig = px.scatter_3d(df, x='x', y='y', z='z',
#               color='ABC-family proteins mediated transport')
# fig.update_traces(marker={'size': 3})
# fig.show()