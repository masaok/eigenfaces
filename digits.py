#!/usr/bin/env python3

# Slide 25
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets as skdata

# We will use the digits dataset
digits_dataset = skdata.load_digits()
X = digits_dataset.data  # (1797, 64)
y = digits_dataset.target

print("Slide 25: Visualize our data")
# Visualize our data
X = np.reshape(X, (-1, 8, 8))  # (1797, 8, 8)
fig = plt.figure()
for i in range(1, 10):
    ax = fig.add_subplot(3, 3, i)
    ax.imshow(X[i, ...])

# plt.show()

plt.show(block=True)  # Show the plot (python3 on Mac terminal)

print("Slide 26: Reshape the image into a vector")
# Reshape the image into a vector (Slide 26)
X = np.reshape(X, (-1, 64))
fig = plt.figure()
for i in range(1, 10):
    ax = fig.add_subplot(9, 1, i)
    ax.imshow(np.expand_dims(X[i, ...], axis=0))

plt.show(block=True)  # Show the plot (python3 on Mac terminal)

# Slide 27
# Before we start applying PCA, let’s split
# the dataset so we can see how good our
# projection (latent vector) is
# Split the data 90-10 training/testing
split_idx = int(0.90*X.shape[0])
X_train, y_train = X[:split_idx, :], X[:split_idx]
X_test, y_test = X[split_idx:, :], X[split_idx:]

print("X_train.shape: " + str(X_train.shape))

# Compute the mean and center the data (Slide 28)
mu_train = np.mean(X_train, axis=0)
B_train = X_train-mu_train

# Compute the covariance matrix
C = np.matmul(B_train.T, B_train)/(B_train.shape[0])  # (64, 1617) x (1617, 64) => (64, 64)

# Eigen decomposition
S, V = np.linalg.eig(C)

# Select the top 3 dimensions
order = np.argsort(S)[::-1]
V = V[:, order]
W = V[:, 0:3]  # Transformation for projecting X to subspace

# Project our data
Z_train = np.matmul(B_train, W)  # (1617, 3)

# Recover our data (Slide 29)
X_train_hat = np.matmul(Z_train, W.T)+mu_train
mse = np.mean((X_train-X_train_hat)**2)  # 11.22242909000862

# Looks like we need more than just 3 dimensions!
# But how many do we need?

# Select the top 45 dimensions (Slide 31)
W = V[:, 0:45]  # Transformation for projecting X to subspace

# Project our data
Z_train = np.matmul(B_train, W)  # (1617, 45)
X_train_hat = np.matmul(Z_train, W.T)+mu_train
mse = np.mean((X_train-X_train_hat)**2)  # 0.07962481763287754

# Looks like we did much better this time

# What if we were to take 10 more eigenvectors? (Slide 32)
W = V[:, 0:55]  # Transformation for projecting X to subspace

# Project our data
Z_train = np.matmul(B_train, W)  # (1617, 55)
X_train_hat = np.matmul(Z_train, W.T)+mu_train
mse = np.mean((X_train-X_train_hat)**2)  # 0.00048743672799306617

# We improve by ~0.06
# Should we increase the number or dimensions?

# Slide 33
# Before we decide to take on more dimensions
# let’s look at the range of values of the images
print(np.min(X), np.max(X))  # The range of intensities in the image between [0, 16]

print("Slide 34: Applying PCA to Handwritten Digits")
# Let’s visualize what our reconstructed digits look like
W = V[:, 0:45]

# Project our data (Slide 34)
Z_train = np.matmul(B_train, W)  # (1617, 55)
X_train_hat = np.matmul(Z_train, W.T)+mu_train

# Visualize our data (Slide 34)
X_train = np.reshape(X_train, (-1, 8, 8))  # because each image has 64 pixels?
X_train_hat = np.reshape(X_train_hat, (-1, 8, 8))
fig = plt.figure()
for i in range(0, 16):  # rendering 16 images total?
    ax = fig.add_subplot(4, 4, i+1)  # 4 x 4 = 16 ?
    if i < 8:
        ax.imshow(X_train[i, ...])
    else:
        ax.imshow(X_train_hat[i-8, ...])

plt.show(block=True)  # Show the plot (python3 on Mac terminal)

# Slide 37
# Let’s try to create novel 9s
idx_9s = np.where(y == 9)[0]
X_9s = X[idx_9s, :]
mu_9s = np.mean(X_9s, axis=0)
B_9s = X_9s-mu_9s

# Covariance matrix
C_9s = np.matmul(B_9s.T, B_9s)/(B_9s.shape[0])  # (64, 64)

# Eigen decomposition
S_9s, V_9s = np.linalg.eig(C_9s)
order_9s = np.argsort(S_9s)[::-1]
W_9s = V_9s[:, order_9s]

print("Slide 38: Generating Novel Imagery")
# Let’s sample from Z to get 9 samples (Slide 38)
Z_9s = np.random.normal(0, np.sqrt(S_9s), (9, S_9s.shape[0]))

# Our novel examples
X_9s_hat = np.matmul(Z_9s, W_9s.T)+mu_9s

# Visualize our data
X_9s_hat = np.reshape(X_9s_hat, (-1, 8, 8))
fig = plt.figure()
for i in range(0, 9):
    ax = fig.add_subplot(3, 3, i+1)
    ax.imshow(X_9s_hat[i, ...])

plt.show(block=True)  # Show the plot (python3 on Mac terminal)
