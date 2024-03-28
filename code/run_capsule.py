""" this script produces all the figures in the summary document, CEBRA Analysis"""
#-----------------------------------------------------------------------
#%%
import sys

import os # my addtion

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
#import joblib as jl
import cebra
import cebra.datasets
from cebra import CEBRA
import torch
from scipy.integrate import solve_ivp

from matplotlib.collections import LineCollection
import pandas as pd

import pickle
import sklearn.linear_model
from sklearn import manifold
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import d_library as dl

#%%
#-----------------------------------------------------------------------
# define results folder
results_folder = r"/results"

# Figure 1
def fig_1():

    #make circles
    circle1 = dl.make_circle()
    circle2 = dl.noisy_circle(a=0.03)
    circles = [circle1, circle2]
    # run function
    dl.r2_vs_dimension(circles,labels=['original circle', 'noisy circle'], dimensionality=[1,2,3,4,5])
    plt.savefig(results_folder + os.sep + "Figure 1" + ".png")

# Figure 2
def fig_2():

    # make attractor
    xyz = dl.make_lorenz()
    # reshape it for use with cebra
    lorenz_obj = xyz.reshape(10000,3)
    lorenz_input = [lorenz_obj]
    #run function
    dl.r2_vs_dimension(lorenz_input,labels=['lorenz attractor'],dimensionality=[1,2,3,4,5])
    plt.savefig(results_folder + os.sep + "Figure 2" + ".png")

# Figure 3
def fig_3():

    # make attractor and compute embedding
    xyz = dl.make_lorenz()
    dl.embed_attractor(xyz)
    print('out of function')
    plt.savefig(results_folder + os.sep + "Figure 3" + ".png")

# Figure 4
def fig_4():

    #make inputs
    circle = dl.make_circle()
    xyz = dl.make_lorenz()
    lorenz_obj = xyz.reshape(10000,3)
    # compute and plot embeddings
    ft_model_c, ft_embedding_c = dl.base_embed(circle, lr=0.1, d=0.5)
    ft_model_l, ft_embedding_l = dl.base_embed(lorenz_obj, lr=0.01, d=0.5)
    dl.plot_2embeddings(ft_embedding_c,ft_embedding_l)
    plt.savefig(results_folder + os.sep + "Figure 4" + ".png")


# Figure 5
def fig_5():

    #make inputs
    circle = dl.make_circle()
    xyz = dl.make_lorenz()
    lorenz_obj = xyz.reshape(10000,3)
    # get their embeddings and plot them
    lt_model_c, lt_embedding_c = dl.base_embed_l(circle, lr=0.1)
    lt_model_l, lt_embedding_l = dl.base_embed_l(lorenz_obj, lr=0.01)
    dl.plot_2embeddings(lt_embedding_c,lt_embedding_l)
    plt.savefig(results_folder + os.sep + "Figure 5" + ".png")


# Figure 6
def fig_6():

    xyz = dl.make_lorenz()
    # reshape it for use with cebra
    lorenz_obj = xyz.reshape(10000,3)
    # run function and save result
    dl.pc_cebra_comp(lorenz_obj, n_iterations=3)
    plt.savefig(results_folder + os.sep + "Figure 6" + ".png")

# Figure 7
def fig_7():

    # make attractor and plot it
    xyz = dl.make_lorenz()
    dl.plot_lorenz(xyz)
    plt.savefig(results_folder + os.sep + "Figure 7" + ".png")


#%%
#-----------------------------------------------------------------------
def run():
    # plot all the figures
    fig_1()
    fig_2()
    fig_3()
    fig_4()
    fig_5()
    fig_6()
    fig_7()
 
if __name__ == "__main__":
    run()
    
#-----------------------------------------------------------------------

# %%
