"""
Script to make mutliple runs on both CEBRA-time and CEBRA-Behaviour modes and compare the reconstruction scores
"""
#%%
#!pip install 'cebra[dev,demos]' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec
import cebra.datasets
from cebra import CEBRA
import d_library as dl
import sklearn.linear_model
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

#-------------------------------------------------------------------
# %%
# Get the data/object
xyz = dl.make_lorenz()
new_lorenz = np.transpose(xyz)

#%%

# time_scores = []
# behaviour_scores = []
label = np.arange(0,new_lorenz.shape[0],1)
# loop over several iterations: 10

# build time model and compute embedding
model,embedding = dl.base_embed(new_lorenz)

# compute reconstruction score
score, transf_embedding = dl.reconstruction_score(embedding, new_lorenz)
# time_scores.append(score)

# do the same for the behaviour model
model_b,embedding_b = dl.base_embed(new_lorenz,b_label=label, mode='delta')
score_b, transf_embedding_b = dl.reconstruction_score(embedding_b, new_lorenz)
# behaviour_scores = []



# iterate then make plot of reconstruction score for both
# %%
