"""
This will act as a library for the dimensionality analysis functions 
"""
import sys

import os # my addtion

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#import joblib as jl
import cebra.datasets
from cebra import CEBRA
import torch
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.collections import LineCollection
import sklearn.linear_model


#--------------------------------------------------------------------
# get r2 score
def reconstruction_score(x, y):

    def _linear_fitting(x, y):
        lin_model = sklearn.linear_model.LinearRegression()
        lin_model.fit(x, y)
        return lin_model.score(x, y), lin_model.predict(x)

    return _linear_fitting(x, y)

#--------------------------------------------------------------------
# function to get an emebdding
def get_embed(input, dimension=3):

    # build CEBRA time model
    model = CEBRA(model_architecture='offset10-model',
                         batch_size=512,
                         learning_rate=0.01,
                         temperature=1,
                         output_dimension = int(dimension),
                         max_iterations=1000,
                         distance='cosine',
                         conditional='time',
                         device='cuda_if_available',
                         num_hidden_units=64,
                         verbose=True,
                         time_offsets=1)

    model.fit(input)
    embedding = model.transform(input)
    return model, embedding
#--------------------------------------------------------------------

# version 2 of get embed -- will align them later: used in pca_cebra_comp
def base_embed(input, temp=1, dimension=3, lr = 0.01, d=0.1):

    # build CEBRA time model
    model = CEBRA(model_architecture='offset1-model',
                         batch_size=512,
                         learning_rate=int(lr),
                         temperature=int(temp),
                         output_dimension = int(dimension),
                         max_iterations=1000,
                         distance='euclidean',
                         delta=int(d),
                         conditional='delta',
                         device='cuda_if_available',
                         num_hidden_units=64,
                         verbose=True,
                         time_offsets=1)

    model.fit(input)
    embedding = model.transform(input)
    return model, embedding
#--------------------------------------------------------------------
# version 3 of get embed: used in Temp analysis for the learnable temp

# function to get an embedding    
def base_embed_l(input, temp=1, dimension=3, lr = 0.01):
# build CEBRA time model
    fixed_temp_model = CEBRA(model_architecture='offset1-model',
                            batch_size=512,
                            learning_rate=int(lr),
                            temperature_mode='auto',
                            min_temperature=0.1,
                            delta=0.5,
                            output_dimension = int(dimension),
                            max_iterations=1000,
                            distance='euclidean',
                            conditional='delta',
                            device='cuda_if_available',
                            num_hidden_units=64,
                            verbose=True,
                            time_offsets=1)

    fixed_temp_model.fit(input)
    embedding = fixed_temp_model.transform(input)
    return fixed_temp_model, embedding


#--------------------------------------------------------------------

# function to get model score + embedding
def get_model_score(data, output_dimension):

    # define a list to hold the scores
    scores =[]

    # get several embeddings and take the average r2 score
    for i in range(0,5):
        # build model and get embedding -- later on edit this to keep the best 1
        model, embedding = get_embed(data, dimension = output_dimension)

        # get r2 score
        model_score, prediction = reconstruction_score(embedding,data)
        scores.append(model_score)

        # conditional to keep the best model here
    
    # get the average
    best = np.mean(scores)
    low = np.min(scores)
    high = np.max(scores)

    model_scores = np.array([best,best-low,high-best])

    return model_scores, embedding

#--------------------------------------------------------------------
# make plots for all the embeddings vs the inputs
def plot_embed_input(input,dimension):

    scores, output = get_model_score(input, dimension)
    ax = cebra.plot_embedding(embedding = output,  markersize=4, alpha=1, embedding_labels='time')
    ax.plot(input[:,0],input[:,1], label='input data')
    ax.legend()

#--------------------------------------------------------------------
# function that gets the r2 scores for input dataset
# take in the dataset - new function definition

def get_r2(dataset, dimensions=[2,3,8]):

    # define list to hold the scores
    r2_scores = []

    # compute the R2 score for multiple dimensions 
    # - call getmodel score for 3 dimensions -- new loop
    for dimension in dimensions:
        r2, output = get_model_score(dataset,int(dimension))
        r2_scores.append(r2)
    
    return np.array(r2_scores)

#--------------------------------------------------------------------
# function that plots R2 vs dimensionality for input dataset
# loop over datasets and get their r2 scores at the 3 dimensions
def r2_vs_dimension(datasets,labels,dimensionality=[2,3,8]):
    # define list to hold the sequence of r2 scores
    r2_sequence =[]

    # loop over all datasets
    for i in range(np.shape(datasets)[0]):
        scores = get_r2(datasets[i],dimensionality)
        r2_sequence.append(scores)

        # then plot the R2 scores at each of the 3 dimensions
        plt.errorbar(x=dimensionality, y=scores[:,0], yerr=np.transpose(scores[:,(1,2)]), fmt='o', label=labels[i])

    plt.xlabel('Dimensionality')
    plt.ylabel('R2 Score')
    plt.title('R2 vs dimensionality')
    plt.legend()
    plt.show()

    return r2_sequence
#--------------------------------------------------------------------

# function to make a circle, that takes in number of data-points
def make_circle(n_points=100, r=1):

    # Define the center and radius of the circle
    center = (0, 0)
    radius = r

    # Create an array of angles from 0 to 2*pi
    theta = np.linspace(0, 2*np.pi, n_points)

    # Calculate the x and y coordinates of the circle
    x = center[0] + radius * np.cos(theta)
    x = x.reshape(-1,1)

    y = center[1] + radius * np.sin(theta)
    y = y.reshape(-1,1)

    # add the 3rd dimension (z coordinate of 0)
    z = np.zeros(n_points)
    z = z.reshape(-1,1)

    circle = np.concatenate((x,y,z), axis=1)

    return circle

#--------------------------------------------------------------------

# function to plot the circle
def plot_circle(x,y,z=0):
    # Create a figure and axis
    fig = plt.figure()
    ax = plt.subplot(111,projection='3d')
    # Plot the circle
    ax.plot(x, y, z, label='Circle')

    # Show the plot
    plt.show()

#--------------------------------------------------------------------

# function that makes the circle and adds the noise and time/theta displacement
def noisy_circle(a=1, b=0.1, n_points=100):

    # tau is uniformly distributed noise between (-0.5, 0.5)
    tau = a*np.random.uniform(-0.5,0.5, size=100)

    # Create an array of angles from 0 to 2*pi
    theta = np.linspace(0, 2*np.pi, n_points)

    # n is zero-mean additive Gaussian noise
    n = np.random.normal(scale=b,size=100)

    # Define the center and radius of the circle
    center = (0, 0)
    radius = 1

    # Calculate the x and y coordinates of the circle with the additive noise
    x = center[0] + radius * np.cos(theta + tau) + n 
    x = x.reshape(-1,1)
    y = center[1] + radius * np.sin(theta + tau) + n
    y = y.reshape(-1,1)

    # add the 3rd dimension (z coordinate of 0)
    z = np.zeros(100) 
    z = z.reshape(-1,1)

    n_circle = np.concatenate((x,y,z), axis=1)

    return n_circle
#--------------------------------------------------------------------
    
# define a function to make a lorenz system
def make_lorenz():    

    # function to define the equations
    def lorenz(t, xyz, sigma=10, rho=28, beta=8/3):
        x, y, z = xyz
        dxdt = sigma * (y - x)
        dydt = rho * x - y - x * z
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    # Define parameters and initial conditions
    sigma = 10
    rho = 28
    beta = 8/3
    initial_conditions = [0, 2, 0]  # Initial conditions for [x, y, z]

    # Time span for integration
    t_span = [0, 35]

    # Solve the differential equations
    solution = solve_ivp(lorenz, t_span, initial_conditions, args=(sigma, rho, beta), dense_output=True)

    # Generate time points for plotting
    t = np.linspace(t_span[0], t_span[1], 10000)

    # Evaluate the solution at the time points
    xyz = solution.sol(t)

    return xyz
#--------------------------------------------------------------------

# Function to plot the Lorenz attractor
def plot_lorenz(coords):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords[0], coords[1], coords[2], alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Attractor')
#--------------------------------------------------------------------

# function to produce embeddings at different timesteps of the attractor
def embed_attractor(attractor):

   # get embeddings in 2D and 3D
   attractor_input = attractor.reshape(attractor.shape[1],attractor.shape[0])
   model_pl2, embed_pl2 = get_embed(attractor_input, dimension=2)
   model_pl3, embed_pl3 = get_embed(attractor_input, dimension=3)

   for i in range(0,attractor.shape[1],1000):

        attractor_p = attractor[:,0:i+1000]

        # define the grid
        gs = gridspec.GridSpec(2, 2, figure=plt.figure(figsize=(10,10)))

        # plot portion of attractor
        ax1 = plt.subplot(gs[0, :], projection='3d')  
        ax1.plot(attractor_p[0], attractor_p[1], attractor_p[2], alpha=1)
        plt.title('Timestep {}'.format(i+1000))

        # plot embeddings
        ax2 = plt.subplot(gs[1,0])
        cebra.plot_embedding(embedding=embed_pl2[0:i+1000,:], embedding_labels='time', markersize=5, ax=ax2, title='2D')
        ax3 = plt.subplot(gs[1,1], projection='3d')
        cebra.plot_embedding(embedding=embed_pl3[0:i+1000,:], embedding_labels='time', markersize=5,ax=ax3, title='3D')
        
        plt.show()

#--------------------------------------------------------------------

# function to compare the top 2 PCs and CEBRA embeddings
def pc_cebra_comp(object, n_iterations = 1):
    
    # Check if the input array has shape (n, m)
    assert object.shape[1] == 3 , f"Input array must have shape 3 columns"

    # define grid
    fig0 = plt.figure(figsize=(8,4*n_iterations))
    gs = gridspec.GridSpec(n_iterations, 2, figure=fig0)

    for i in range(0,n_iterations):

        # make a pca model and fit on the object
        pca = PCA(n_components=2)
        pca.fit(object)

        # compute the PCs and get the explained variance
        object_pca = pca.fit_transform(object)
        explained_var = pca.explained_variance_ratio_

        # compute the cebra embedding using the ideal params learned earlier
        cebra_model, cebra_embedding = base_embed(input=object, temp=1,dimension=3, lr=0.01, d=1)
        
        # make a plot of the 2 PCs of the object
        ax0 = fig0.add_subplot(gs[i,0], projection='3d')
        ax0.scatter(object_pca[:,0], object_pca[:,1], s=0.07)
        ax0.set_title('Top 2 PCs of Lorenz Attractor')

        # plot the cebra embedding
        ax1 = fig0.add_subplot(gs[i,1], projection='3d') 
        cebra.plot_embedding(embedding=cebra_embedding, embedding_labels='time',ax=ax1, markersize=0.001, alpha=1, title='Lorenz Attractor Embedding')
        plt.show()

    return explained_var
# plot the embeddings
def plot_2embeddings(embed1,embed2):
    fig0 = plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(1, 2, figure=fig0)

    ax0 = fig0.add_subplot(gs[0,0], projection='3d')
    ax1 = fig0.add_subplot(gs[0,1], projection='3d')
    cebra.plot_embedding(embed1, embedding_labels='time', ax=ax0, markersize=5, alpha=1, title='Circle')
    cebra.plot_embedding(embed2, embedding_labels='time',ax=ax1, markersize=0.001, alpha=1, title='Lorenz Attractor')
    plt.show()
#--------------------------------------------------------------------