"""
This is a script to produce some figures ( 2D CEBRA embeddings) for a presentation

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
n_iterations = 2

for i in range(0,n_iterations):

    # generate embeddings
    cebra_model2, cebra_embedding2 = dl.base_embed(input=new_lorenz, temp=0.001,dimension=2, lr=0.01, d=0.1)
    cebra_score2, transformed_embedding2 = dl.reconstruction_score(cebra_embedding2,new_lorenz)

    # keep the best emebdding
    if cebra_score2 > best_score:
        best_score = cebra_score2
        best_embedding = cebra_embedding2
        best_t_embedding = transformed_embedding2

    # keep track of the scores so we can calculate the mean score after
    scores.append(cebra_score2)
    
# get the mean score
mean_score = np.mean(scores)

# plot the best embedding and save it
fig_new = plt.figure()
plt.plot(new_lorenz[:,0], new_lorenz[:,1])
plt.title('Lorenz Attractor')

fig2_new = plt.figure()
ax_new = fig2_new.add_subplot()
cebra.plot_embedding(embedding=best_embedding, embedding_labels='time',ax=ax_new,cmap='cool', markersize=0.01, alpha=1, title='CEBRA Embedding')
