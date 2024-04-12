# ECE 1155 Term Paper Simulation
# Anomaly detection techniques

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerPathCollection
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import median_abs_deviation


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
X = np.r_[X_test]
clf = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
y_pred = clf.fit_predict(X)
X_scores = clf.negative_outlier_factor_
plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")

# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)
plt.axis("auto")

#plt.legend(
 #   handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
#)
plt.title("Local Outlier Factor (LOF)")
plt.savefig('output/LOF.png')


#----------------------------------------
#    Double Median Absolute Deviation - John Darnall
#----------------------------------------

#calculating double median absolute deviation bounds
k = 3 #default standard check
median = np.median(X_test)

lX_test = X_test[X_test <= median]
uX_test = X_test[X_test > median]

lMAD = median_abs_deviation(lX_test)
uMAD = median_abs_deviation(uX_test)

lower = median - k*lMAD
upper = median + k*uMAD

#comparing each point to upper and lower bounds to check for anomalies
low_anomalies_indices = np.where(X_test < lower)[0]
high_anomalies_indices = np.where(X_test >= upper)[0]
anomalies_indices = np.concatenate((low_anomalies_indices, high_anomalies_indices))
anomalies = X_test[anomalies_indices]

print("Dataset: ", X_test)
print("Anomalies: ", anomalies)

#plotting data points
plt.figure()
plt.scatter(X_test[:,0], X_test[:,1], color = 'blue', label = 'Benign Data')
plt.scatter(anomalies[:,0], anomalies[:,1], color = 'red', label = 'Anomalies')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.title('Anomaly Detection Using Double Median Absolute Deviation')
plt.legend()
plt.savefig('output/DoubleMAD.png')

#standard MAD comparison
MAD = median_abs_deviation(X_test)
tlower = median - k*MAD
tupper = median + k*MAD

tlow_anomalies_indices = np.where(X_test < tlower)[0]
thigh_anomalies_indices = np.where(X_test >= tupper)[0]
tanomalies_indices = np.concatenate((tlow_anomalies_indices, thigh_anomalies_indices))
tanomalies = X_test[tanomalies_indices]

print("Dataset: ", X_test)
print("Anomalies: ", tanomalies)

#plotting data points
plt.figure()
plt.scatter(X_test[:,0], X_test[:,1], color = 'blue', label = 'Benign Data')
plt.scatter(tanomalies[:,0], tanomalies[:,1], color = 'red', label = 'Anomalies')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.title('Anomaly Detection Using Standard Median Absolute Deviation')
plt.legend()
plt.savefig('output/StandardMAD.png')


#----------------------------------------
#           Isolation Forest
#----------------------------------------
