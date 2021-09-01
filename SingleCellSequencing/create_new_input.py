# take the input from R script we are looking at SNU-668/Clone_0.00236009_ID107574 cell 5 here pathway 7 
# I want to plot proportions but it is easier for me in python

import numpy as np 
import pandas as pd

p_expr = pd.read_csv("/Users/esthomas/Downloads/example_pathway_expr.csv")
idx = pd.read_csv("/Users/esthomas/Downloads/example_idx_Candid.csv")
#fr = pd.read_csv("/Users/esthomas/Downloads/example_fr.csv")
coord = pd.read_csv("/Users/esthomas/Downloads/example_coord.csv")
mito = pd.read_csv("/Users/esthomas/Downloads/mito.csv")
nucl = pd.read_csv("/Users/esthomas/Downloads/nucl.csv")

total_expr = sum(p_expr["x"][0:100])

# if I just look at expression in the nucleus for now

dataframe = pd.DataFrame(zip(p_expr["x"][0:100], mito["present"], nucl["present"]), columns = ["expr", "mito", "nucl"])

just_mito = dataframe.loc[(dataframe['mito'] == 1) & (dataframe['nucl'] == 0), 'expr'].sum()
just_nucl = dataframe.loc[(dataframe['mito'] == 0) & (dataframe['nucl'] == 1), 'expr'].sum()
both = dataframe.loc[(dataframe['mito'] == 1) & (dataframe['nucl'] == 1), 'expr'].sum()

total_nucl = just_nucl + 0.5*both
total_mito = just_mito + 0.5*both

print(total_nucl, total_mito, total_nucl + total_mito)

pathway_n = []
pathway_m = []
expr_n = []
expr_m = []


for name, p, m, n in zip(p_expr["pathway"][0:100], p_expr["x"][0:100], mito["present"], nucl["present"]):
    if n == 1 and m == 0:
        per = (p/total_nucl)*100
        points = (total_nucl/100) * per
        pathway_n.append(name)
        expr_n.append(round(points))
    elif n == 0 and m == 1: 
        per = (p/total_mito)*100
        points = (total_mito/100) * per
        pathway_m.append(name)
        expr_m.append(round(points))
    elif n == 1 and m == 1:
        per = ((p/2)/total_mito)*100
        points = (total_mito/100) * per
        pathway_m.append(name)
        expr_m.append(round(points))

        per = ((p/2)/total_nucl)*100
        points = (total_nucl/100) * per
        pathway_n.append(name)
        expr_n.append(round(points))
    else:
        print("not matched conditions")

print(sum(expr_n), total_nucl)

print(len(expr_n))
print(len(expr_m))

nuc_coords = coord[coord["nucleus"] == 1]
nuc_coords = nuc_coords[["x", "y", "z"]]
print(nuc_coords)

n_array = nuc_coords.to_numpy()
print(n_array)

from sklearn.neighbors import KDTree

# this is just for a dummy pathway pretending no. of points is 200
# ideally i should really start  
tree = KDTree(n_array)
print(n_array[0])
nearest_dist, nearest_ind = tree.query([n_array[0]], k=200)

print(nearest_ind[0])

nuc_coords["colour"] = np.zeros
nuc_coords.loc[nearest_ind[0],'colour'] = 1

new_n_array = np.delete(n_array, nearest_ind[0], 0)

import plotly.express as px

fig = px.scatter_3d(nuc_coords, x='x', y='y', z='z',
              colour="colours")
fig.show()






