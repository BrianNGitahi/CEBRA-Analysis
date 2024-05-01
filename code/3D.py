"""
This is a script to produce some figures ( 3D CEBRA embeddings) for a presentation

"""

#! pip install 'cebra[dev,demos]'
#%%
import sys
import os # my addtion

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import cebra.datasets
from cebra import CEBRA
import torch

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score


from matplotlib.collections import LineCollection
import pandas as pd
import d_library as dl


#%%
# make the lorenz attractor
xyz = dl.make_lorenz()
new_lorenz = np.transpose(xyz)

# best embedding
best_embedding = None
best_t_embedding = None

# keep track of the best reconstruction score
best_score = 0

# store the reconstruction scores
scores = []


# run CEBRA multiple times on the attractor and save the embeddings
n_iterations = 10
for i in range(0,n_iterations):

    # generate the embedding 
    cebra_model, cebra_embedding = dl.base_embed(input=new_lorenz, temp=0.001,dimension=3, lr=0.01, d=1)
    cebra_score, transformed_embedding = dl.reconstruction_score(cebra_embedding,new_lorenz)

    # keep the best embedding
    if cebra_score > best_score:
        best_score = cebra_score
        best_embedding = cebra_embedding
        best_t_embedding = transformed_embedding

    # keep track of the scores so we can calculate the mean score after
    scores.append(cebra_score)

#%%
# get the mean score
mean_score = np.mean(scores)

# plot the best embedding and save it

# define grid
fig0 = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(2, 2, figure=fig0)


# make a plot of the 2 PCs of the object
ax0 = fig0.add_subplot(gs[0,0], projection='3d')
ax0.scatter(new_lorenz[:,0], new_lorenz[:,1],new_lorenz[:,2], s=0.07)
ax0.set_title('Lorenz Attractor')

# plot the cebra embedding
ax1 = fig0.add_subplot(gs[0,1], projection='3d')         
cebra.plot_embedding(embedding=best_embedding, embedding_labels='time',ax=ax1,cmap='cool', markersize=0.001, alpha=1, title='CEBRA Embedding')

ax2 = fig0.add_subplot(gs[1,:], projection='3d')
cebra.plot_embedding(embedding=best_t_embedding, embedding_labels='time',ax=ax2,cmap='cool', markersize=0.001, alpha=1, title='Transformed CEBRA Embedding, mean:{}'.format(np.round(mean_score, 3)))

plt.savefig('3D Embedding of lorenz.png')

print('Best reconstruction score: {}'.format(best_score))


