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
def r2_vs_dimension(datasets,labels, dimensionality=[2,3,8], ):
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