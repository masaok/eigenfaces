#!/usr/bin/env python3

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

import argparse
import logging

from pprint import pprint
from pprint import pformat

import re

import os.path

"""
Name: Kitamura, Masao

Collaborators: Huerta, Emilia

Collaboration details: Discussed <function name> implementation details with Jane Doe.
"""

CELEBA_DIRPATH = 'celebA_dataset'
N_HEIGHT = 78
N_WIDTH = 78
N_TRAIN = 850

# Parse command line options
parser = argparse.ArgumentParser(description='Do stuff.')
parser.add_argument(
    '-d', '--debug', dest='debug', action='store_true', default=0, help='debug mode')
parser.add_argument(
    '-i', '--info', dest='info', action='store_true', default=0, help='info mode')
parser.add_argument(
    '-w', '--warn', dest='warn', action='store_true', default=0, help='warn mode')

args = parser.parse_args()

# Initialize logging
logging.basicConfig(format='%(levelname)s [%(filename)s:%(lineno)4d %(funcName)s] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

log = logging.getLogger(__name__)
if args.debug:
    log.setLevel(logging.DEBUG)
elif args.info:
    log.setLevel(logging.INFO)
elif args.warn:
    log.setLevel(logging.WARNING)


def get_eigenfaces(eigenvalues, eigenvectors, k):
    """
      Sorts the eigenvector by eigenvalues
      Returns the projection matrix (eigenfaces)

      faces_centered : N x d vector
        mean subtracted faces
      eigenvalues : 1 x d vector
        eigenvalues
      eigenvectors : d x d vector
        eigenvectors
      k : int
        number of eigenfaces to produce

      returns d x k vector
    """

    # sort eigenvalues
    # take order
    # sort eigenvectors
    # select from 0 to k
    # eigenvectors
    # return eigenfaces ?


def project(faces, faces_mean, eigenfaces):
    """
      Returns the projected faces (lower dimensionality)

      faces : N x d vector
        images of faces
      faces_mean : 1 x d vector
        per pixel average of images of faces
      eigenfaces : d x k vector
        projection matrix

      returns N x k vector
    """


def reconstruct(faces_projected, faces_mean, eigenfaces):
    """
      Returns the reconstructed faces (back-projected)

      faces_projected : N x k vector
        faces in lower dimensions
      faces_mean : 1 x d vector
        per pixel average of images of faces
      eigenfaces : d x k vector
        projection matrix

      returns N x d vector
    """


def synthesize(eigenfaces, variances, faces_mean, k=50, n=25):
    """
      Synthesizes new faces by sampling from the latent vector distribution

      eigenfaces : d x k vector
        projection matrix
      variances : 1 x d vector
        variances
      faces_mean : 1 x d vector

      returns synthesized faces
    """


def mean_squared_error(x, x_hat):
    """
      Computes the mean squared error

      x : N x d vector
      x_hat : N x d vector

      returns mean squared error
    """


def plot_eigenvalues(eigenvalues):
    """
      Plots the eigenvalues from largest to smallest

      eigenvalues : 1 x d vector
    """
    fig = plt.figure()
    fig.suptitle('Eigenvalues Versus Principle Components')


def visualize_reconstructions(faces, faces_hat, n=4):
    """
      Creates a plot of 2 rows by n columns
      Top row should show original faces
      Bottom row should show reconstructed faces (faces_hat)

      faces : N x k vector
        images of faces
      faces_hat : 1 x d vector
        images reconstructed faces
    """
    fig = plt.figure()
    fig.suptitle('Real Versus Reconstructed Faces')


def plot_reconstruction_error(mses, k):
    """
      Plots the reconstruction errors

      mses : list
        list of mean squared errors
      ks : list
        list of k used
    """
    fig = plt.figure()
    fig.suptitle('Reconstruction Error')


def visualize_eigenfaces(eigenfaces):
    """
      Creates a plot of 5 rows by 5 columns
      Shows the first 25 eigenfaces (principal components)
      For each dimension k, plot the d number values as an image

      eigenfaces : d x k vector
    """
    fig = plt.figure()
    fig.suptitle('Top 25 Eigenfaces')


def visualize_synthetic_faces(faces):
    """
      Creates a plot of 5 rows by 5 columns
      Shows the first 25 synthetic faces

      eigenfaces : N x d vector
    """
    fig = plt.figure()
    fig.suptitle('Synthetic Faces')


if __name__ == '__main__':
    # Load faces from directory
    face_image_paths = glob.glob(os.path.join(CELEBA_DIRPATH, '*.jpg'))

    print('Loading {} images from {}'.format(len(face_image_paths), CELEBA_DIRPATH))
    # Read images as grayscale and resize from (128, 128) to (78, 78)
    faces = []
    for path in face_image_paths:
        im = Image.open(path).convert('L').resize((N_WIDTH, N_HEIGHT))
        faces.append(np.asarray(im))
    faces = np.asarray(faces)  # (1000, 78, 78)
    # Normalize faces between 0 and 1
    faces = faces/255.0

    log.info('Vectorizing faces into N x d matrix')
    # DONE: Reshape the faces to into an N x d matrix (slide 26)
    log.info(faces)
    log.info(faces.shape)  # (1000, 78, 78)
    faces = np.reshape(faces, (-1, 6084))  # 78 * 78 = 6084

    log.info('Splitting dataset into {} for training and {} for testing'.format(
        N_TRAIN, faces.shape[0]-N_TRAIN))
    faces_train = faces[0:N_TRAIN, ...]
    faces_test = faces[N_TRAIN::, ...]

    X_train = faces_train

    log.info('Computing eigenfaces from training set (slide 28)')
    # DONE: obtain eigenfaces and eigenvalues
    mu_train = np.mean(faces_train, axis=0)
    log.info("mu_train.shape: " + str(mu_train.shape))

    B_train = faces_train - mu_train
    log.info("B_train.shape: " + str(B_train.shape))
    log.info("B_train.shape[0]: " + str(B_train.shape[0]))

    # Find the covariance matrix (slide 28)
    # (850, 6084) x (6084, 850) => (6084, 6084)
    C = np.matmul(B_train.T, B_train)/(B_train.shape[0])
    log.info("C.shape: " + str(C.shape))

    # Cache locally
    sigmafile = "data_sigma.npy"
    vectorfile = "data_vectors.npy"

    if os.path.isfile(sigmafile) and os.path.isfile(vectorfile):
        log.info("Cache files exist")
        S = np.load(sigmafile)
        V = np.load(vectorfile)
    else:
        log.info("Cache files do not exist")
        # Eigen decomposition
        S, V = np.linalg.eig(C)
        np.save(sigmafile, S)
        np.save(vectorfile, V)

    log.info('Plotting the eigenvalues from largest to smallest')
    # DONE: plot the first 200 eigenvalues from largest to smallest

    #fig = plt.figure()
    # for i in range(1, 200):
    #ax = fig.add_subplot()

    log.info(S)
    log.info(V)

    log.info("S.shape: " + str(S.shape))  # (6084,)
    log.info("V.shape: " + str(V.shape))  # (6084, 6084)

    # Sort the values in descending order
    top_values = np.sort(S)
    top_values = np.flip(top_values)

    # # Another way to sort?
    # order = np.argsort(S)[::-1]
    # V = V[:, order]

    # Get the top 200
    top_values = S[0:200]

    # Alternate way?
    # W = V[:, 0:200]  # Transformation for projecting X to subspace

    # Project our data
    # Z_train = np.matmul(B_train, W)  # (850, 200)
    # log.info("Z_train.shape: " + str(Z_train.shape))

    # X_train_hat = np.matmul(Z_train, W.T)+mu_train  # (850, 6084)
    # log.info("X_train_hat.shape: " + str(X_train_hat.shape))

    # X_train = faces_train
    # mse = np.mean((X_train-X_train_hat)**2)
    # log.info("MSE: " + str(mse))

    # Plot and show
    log.info("top_values.shape: " + str(top_values.shape))
    plt.plot(top_values)
    # plt.plot(W)  # Other way, as done in digits example?
    plt.show(block=True)  # Show the plot (python3 on Mac terminal)

    log.info('Visualizing the top 25 eigenfaces')

    log.info(matplotlib.get_backend()) # debug

    # DONE: visualize the top 25 eigenfaces
    log.info("faces_train.shape: " + str(faces_train.shape))  # (850, 6048) and sqrt(6084) = 78
    faces_train = np.reshape(faces_train, (-1, 78, 78))
    fig = plt.figure()
    fig.suptitle('Top 25 Eigenfaces')
    for i in range(0, 25):
        ax = fig.add_subplot(5, 5, i+1)

        # log.info("faces_train[i, ...].shape: " + str(faces_train[i, ...].shape))
        ax.imshow(faces_train[i, ...])

    plt.show(block=True)

    log.info('Plotting training reconstruction error for various k')
    # TODO: plot the mean squared error of training set with
    # k=[5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]


    W = V[:, 0:5]
    Z_train = np.matmul(B_train, W)
    X_train_hat = np.matmul(Z_train, W.T)+mu_train
    mse = np.mean((X_train-X_train_hat)**2)

    log.info("mse: " + str(mse))


    print('Reconstructing faces from projected faces in training set')
    # TODO: choose k and reconstruct training set

    # TODO: visualize the reconstructed faces from training set

    print('Reconstructing faces from projected faces in testing set')
    # TODO: reconstruct faces from the projected faces

    # TODO: visualize the reconstructed faces from training set

    print('Plotting testing reconstruction error for various k')
    # TODO: plot the mean squared error of testing set with
    # k=[5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]

    print('Creating synethic faces')
    # TODO: synthesize and visualize new faces based on the distribution of the latent variables
    # you may choose another k that you find suitable
