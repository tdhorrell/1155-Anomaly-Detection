# ECE 1155 Term Paper Simulation
# Anomaly detection techniques

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# generate random 2d data using numpy
np.random.seed(7)
mu, sigma = 0, 0.1

# Gaussian distributed standard data
X_data = np.random.normal(mu, sigma, (1000, 2))
y_data = np.zeros((1000, 1))

# Uniformly distributed outliers
X_outliers = np.random.uniform(low=-3, high=3, size=(50,2))
y_outliers = np.ones((50, 1))

# stack the data into X and y arrays with labels
anomaly_data_X = np.vstack((X_data, X_outliers))
anomaly_data_y = np.vstack((y_data, y_outliers))

# separate the data into testing and training
X_train, X_test, y_train, y_test = train_test_split(anomaly_data_X, anomaly_data_y, test_size=0.2, random_state=7)

# NOTE: Anomaly detection needs to be robust to new data. 
# The train_test_split data has outliers present and as a result would need trained with labels a.k.a. y_values
# In unspervised methods, the unlabeled X_data will suffice for training.