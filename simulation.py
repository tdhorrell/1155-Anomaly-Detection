# ECE 1155 Term Paper Simulation
# Anomaly detection techniques

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# generate random 2d data using numpy
np.random.seed(7)
mu, sigma = 0, 0.5

# Gaussian distributed standard data
X_data = np.random.normal(mu, sigma, (1000, 2))
y_data = np.zeros((1000, 1))

# Uniformly distributed outliers
X_outliers = np.random.uniform(low=-3, high=3, size=(50,2))
y_outliers = np.ones((50, 1))

# stack the data into X and y arrays with labels
anomaly_data_X = np.vstack((X_data, X_outliers))
anomaly_data_y = np.vstack((y_data, y_outliers)).squeeze(1)

# separate the data into testing and training
X_train, X_test, y_train, y_test = train_test_split(anomaly_data_X, anomaly_data_y, test_size=0.2, random_state=7)

# NOTE: Anomaly detection needs to be robust to new data. 
# The train_test_split data has outliers present and as a result would need trained with labels a.k.a. y_values
# In unspervised methods, the unlabeled X_data will suffice for training.

# plot to visualize data
plt.scatter(X_data[:, 0], X_data[:, 1], c='blue')
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
plt.xlabel("x0")
plt.ylabel("x1")
plt.legend(['Benign Data', 'Outliers'], loc='upper right')
plt.title("Anomaly Dataset")
plt.savefig("output/anomaly_dataset.png")

# set colormaps for figures
cm = plt.set_cmap('RdBu')
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

#----------------------------------------
#   MLP ANN Classifier - Tim Horrell
#----------------------------------------

# The MLP Network utilizes stochastic gradient descent (Adam, Relu, (100,) hidden layer defaults)
# Fit the Neural Network
L2_regularization = 0.0001
mlp = MLPClassifier(alpha=L2_regularization, random_state=7, max_iter=500).fit(X_train, y_train)

# accuracy test
print(f'MLP Network Accuracy: ',mlp.score(X_test, y_test))
# NOTE: Accuracy is historically 99%

# view decision boundary
MLP_figure = plt.figure()
mlp_ax = plt.subplot(1, 1, 1)
mlp_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap = cm_bright, edgecolors='k')
mlp_ax.set_title("MLP Test Data and Decision Boundary")
DecisionBoundaryDisplay.from_estimator(mlp, X_test, cmap=cm, alpha=0.8, ax=mlp_ax, eps=5)
plt.savefig('output/MLP_test_decision_boundary.png')

#----------------------------------------
#       Local Outlier Technique
#----------------------------------------



#----------------------------------------
#    Double Median Absolute Distance
#----------------------------------------



#----------------------------------------
#           Isolation Forest
#----------------------------------------
