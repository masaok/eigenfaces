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
               Wong, Alex
               Abdelalim

Collaboration details: Discussed all implementation details with all collaborators.

Notes:

Three use cases of PCA:

1. Compression/decompression
2. Synthesizing new faces
3. Removing dimensions that are not useful
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
    order = np.argsort(-eigenvalues)
    log.info("order: " + str(order))

    # take order
    values = eigenvalues[order]
    log.info("values: " + str(values))

    # sort eigenvectors
    vectors = eigenvectors[:, order]
    # log.info("vectors: " + str(vectors))

    # select from 0 to k
    eigenfaces = vectors[:, 0:k].real

    return eigenfaces


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
    # Center the faces
    B = faces - faces_mean

    # Multiply centered faces with eigenfaces
    return np.matmul(B, eigenfaces)


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
    return np.matmul(faces_projected, eigenfaces.T) + faces_mean


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
    log.info("variances.shape: " + str(variances.shape))

    Z = np.random.normal(0,
                         np.sqrt(variances[0:k]),
                         (n, variances[0:k].shape[0]))

    log.info("Z.shape: " + str(Z.shape))
    log.info("eigenfaces.T.shape: " + str(eigenfaces.T.shape))

    X_hat = np.matmul(Z, eigenfaces.T) + faces_mean
    return X_hat


def mean_squared_error(x, x_hat):
    """
      Computes the mean squared error

      x : N x d vector
      x_hat : N x d vector

      returns mean squared error
    """
    return np.mean((x - x_hat)**2)


def plot_eigenvalues(eigenvalues):
    """
      Plots the eigenvalues from largest to smallest

      eigenvalues : 1 x d vector
    """
    fig = plt.figure()
    fig.suptitle('Eigenvalues Versus Principle Components')
    plt.xlabel('Principle Components')
    plt.ylabel('Eigenvalues')
    plt.plot(eigenvalues)
    plt.show(block=True)


def visualize_reconstructions(faces, faces_hat, label="", n=4):
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
    fig.suptitle('Real Versus Reconstructed Faces: ' + label)

    for i in range(0, n * 2):
        ax = fig.add_subplot(2, n, i+1)
        if i < n:
            ax.imshow(faces[i, ...], cmap='gray')
        else:
            ax.imshow(faces_hat[i-n, ...], cmap='gray')

    plt.show(block=True)


def plot_reconstruction_error(mses, ks, name):
    """
      Plots the reconstruction errors

      mses : list
        list of mean squared errors
      ks : list
        list of k used
    """
    plt.title('Reconstruction Error: ' + str(name))
    plt.xlabel('K values')
    plt.ylabel('MSE')
    plt.plot(ks, mses)
    plt.show(block=True)


def visualize_eigenfaces(eigenfaces):
    """
      Creates a plot of 5 rows by 5 columns
      Shows the first 25 eigenfaces (principal components)
      For each dimension k, plot the d number values as an image

      eigenfaces : d x k vector
    """
    fig = plt.figure()
    fig.suptitle('Top 25 Eigenfaces')

    k = 25
    for i in range(0, k):
        ax = fig.add_subplot(5, 5, i+1)
        ax.imshow(eigenfaces[i, ...], cmap='gray')

    plt.show(block=True)


def visualize_synthetic_faces(faces):
    """
      Creates a plot of 5 rows by 5 columns
      Shows the first 25 synthetic faces

      eigenfaces : N x d vector
    """
    fig = plt.figure()
    fig.suptitle('Synthetic Faces')

    for i in range(0, faces.shape[0]):
        ax = fig.add_subplot(5, 5, i+1)
        ax.imshow(faces[i, ...], cmap='gray')

    plt.show(block=True)


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

    log.info("faces_train.shape: " + str(faces_train.shape))
    log.info("faces_test.shape: " + str(faces_test.shape))

    X_train = faces_train
    X_test = faces_test

    log.info('Computing eigenfaces from training set (slide 28)')
    # DONE: obtain eigenfaces and eigenvalues
    mu_train = np.mean(faces_train, axis=0)
    log.info("mu_train.shape: " + str(mu_train.shape))

    B_train = faces_train - mu_train
    log.info("B_train.shape: " + str(B_train.shape))
    log.info("B_train.shape[0]: " + str(B_train.shape[0]))

    log.info("np.matmul(B_train.T, B_train).shape: " + str(np.matmul(B_train.T, B_train).shape))

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

    log.info("S.shape: " + str(S.shape))  # (6084,)
    log.info("V.shape: " + str(V.shape))  # (6084, 6084)

    # Sort the values in descending order
    top_values = np.sort(S)
    top_values = np.flip(top_values)

    # Get the top 200
    k = 200
    top_values = S[0:k]

    # Plot and show
    plot_eigenvalues(top_values)

    log.info('Visualizing the top 25 eigenfaces')

    # Get eigenfaces based on decomposition and k
    # Eigenfaces are the eigenvectors for the faces set
    eigenfaces = get_eigenfaces(S, V, k)
    log.info("eigenfaces.shape: " + str(eigenfaces.shape))

    eigenfaces = eigenfaces.T
    log.info("eigenfaces.T.shape: " + str(eigenfaces.shape))

    eigenfaces = np.reshape(eigenfaces, (-1, 78, 78))
    log.info("eigenfaces.shape after reshape: " + str(eigenfaces.shape))

    visualize_eigenfaces(eigenfaces)

    log.info('Plotting training reconstruction error for various k')
    # DONE: plot the mean squared error of training set with

    k_values = [5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]
    errors = []

    for k in k_values:
        log.info('k: ' + str(k))
        W = V[:, 0:k]  # (6084, k)

        # Project B to Z
        # Z_train = np.matmul(B_train, W)  # (850, k)
        Z_train = project(faces_train, mu_train, W)  # (850, k)

        # Recontruct Z to X_hat
        # X_train_hat = np.matmul(Z_train, W.T) + mu_train  # (850, 6084)
        X_train_hat = reconstruct(Z_train, mu_train, W)  # (850, 6084)

        # Measure loss
        mse = mean_squared_error(X_train, X_train_hat)

        # Append to array
        errors.append(mse)

    log.info("errors: " + str(errors))
    plt.title("Mean Squared Error of training set with various K values")
    plt.xlabel('K values')
    plt.ylabel('MSE')
    plt.plot(k_values, errors)
    plt.show(block=True)  # Show MSE plot

    print('Reconstructing faces from projected faces in training set')
    # TODO: choose k and reconstruct training set

    # Z is projected faces, and Z = BW
    # W is eigenfaces with dimensions d x k
    # B is centered faces (N x d) X (d x k) = N x k
    # Therefore, Z has fewer dimensions
    # Hence, B was projected to Z

    k = 50

    W = V[:, 0:k].real  # converts complex128 to float64 (needed for imshow to work later)
    log.info("W.shape: " + str(W.shape))
    log.info("W.dtype: " + str(W.dtype))

    # Project B to Z (compress)
    Z_train = np.matmul(B_train, W)  # (850, 50)
    log.info("Z_train.shape: " + str(Z_train.shape))

    # Reconstruct (decompress)
    X_train_hat = np.matmul(Z_train, W.T) + mu_train
    log.info("X_train_hat.shape: " + str(X_train_hat.shape))  # (850, 6084)

    X_train_hat = np.reshape(X_train_hat, (-1, 78, 78))
    log.info("X_train_hat.shape: " + str(X_train_hat.shape))  # (850, 78, 78)
    log.info("X_train_hat.dtype: " + str(X_train_hat.dtype))

    X_train_faces = np.reshape(X_train, (-1, 78, 78))

    # DONE: visualize the reconstructed faces from training set
    visualize_reconstructions(X_train_faces, X_train_hat, "training set")

    print('Reconstructing faces from projected faces in testing set')
    # TODO: reconstruct faces from the projected faces

    # mu_test = np.mean(X_test, axis=0)

    B_test = X_test - mu_train  # According to Slack chat with Dr. Wong
    C = np.matmul(B_test.T, B_test)/(B_test.shape[0])

    # Project B to Z (compress)
    Z_test = np.matmul(B_test, W)  # (850, 50)
    log.info("Z_test.shape: " + str(Z_test.shape))

    # Reconstruct (decompress)
    X_test_hat = np.matmul(Z_test, W.T) + mu_train
    log.info("X_test_hat.shape: " + str(X_test_hat.shape))  # (850, 6084)

    X_test_hat = np.reshape(X_test_hat, (-1, 78, 78))
    log.info("X_test_hat.shape: " + str(X_test_hat.shape))  # (850, 78, 78)
    log.info("X_test_hat.dtype: " + str(X_test_hat.dtype))

    X_test_faces = np.reshape(X_test, (-1, 78, 78))

    # TODO: visualize the reconstructed faces from testing set
    visualize_reconstructions(X_test_faces, X_test_hat, "testing set")

    print('Plotting testing reconstruction error for various k')
    # TODO: plot the mean squared error of testing set with
    k_values = [5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]

    mses = []
    for k in k_values:
        W = V[:, 0:k]  # (6084, k)

        z_k = np.matmul(B_test, W)  # (150, k)

        x_hat_k = np.matmul(z_k, W.T) + mu_train
        mse = mean_squared_error(X_test, x_hat_k)
        mses.append(mse)

    plot_reconstruction_error(mses, k_values, "testing set")

    print('Creating synthetic faces (Slide 38)')
    # TODO: synthesize and visualize new faces based on the distribution of the latent variables
    # you may choose another k that you find suitable

    k = 50
    eigenfaces = get_eigenfaces(S, V, k)
    variances = S.real
    faces_mean = mu_train
    X_hat = synthesize(eigenfaces, variances, faces_mean)
    log.info("X_hat.shape: " + str(X_hat.shape))

    X_hat = np.reshape(X_hat, (-1, 78, 78))
    visualize_synthetic_faces(X_hat)
