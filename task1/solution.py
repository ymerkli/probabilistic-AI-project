# Standard scientific Python imports
import numpy as np
import pickle
from time import time

import sklearn.gaussian_process as gp
import sklearn.gaussian_process.kernels as ker

from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, WhiteKernel
from sklearn.preprocessing import StandardScaler



## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04

def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true >= THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted > THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<=THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true < THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    #reward = W4*np.logical_and(predicted <= THRESHOLD,true<=THRESHOLD)
    cost2 = W4*(np.logical_and(predicted >= THRESHOLD,true<=THRESHOLD).astype(int)
    - np.logical_and(predicted <= THRESHOLD,true<=THRESHOLD).astype(int))
    if cost2 is None:
        cost2 = 0

    return np.mean(cost) + np.mean(cost2)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model():

    def __init__(self):
        """
            TODO: enter your code here
        """
        pass

    def preprocess(self, train_x, train_y, num_dense_samples):
        """
            Data is heavily imbalaced, there is a lot of datapoints (17100) in the [-1, -0.5] x0 region
            and only few datapoints (150) in the [-0.5, 1] x0 region.

            We subsample the dense region to create a balanced dataset and at the same time deal with
            the large scale of the dataset.

            Args:
                train_x (numpy.array):  training data points
                train_y (numpy.array):  training labels
                num_dense_labels (int): number of dense data points that should be subsampled
            Returns:
                train_x_s (numpy.array):    uniformly sampled and balanced training data
                train_y_s (numpy.array):    corresponding training labels
        """
        # separate all the valuable datapoints
        train_x_sparse = train_x[train_x[:,0] > -0.5]
        train_y_sparse = train_y[train_x[:,0] > -0.5]

        train_x_dense = train_x[train_x[:,0] <= -0.5]
        train_y_dense = train_y[train_x[:,0] <= -0.5]

        dense_data = np.concatenate([train_x_dense, train_y_dense.reshape(-1, 1)], axis=1)
        np.random.shuffle(dense_data)

        # random shuffle datapoints in the dense region and select 150 to balance the data
        train_x_dense_sampled = dense_data[:num_dense_samples,:2]
        train_y_dense_sampled = dense_data[:num_dense_samples,2]

        train_x_s = np.concatenate([train_x_sparse, train_x_dense_sampled], axis=0)
        train_y_s = np.concatenate([train_y_sparse, train_y_dense_sampled])

        assert train_x_s.shape[0] == num_dense_samples + train_x_sparse.shape[0]
        assert train_y_s.shape[0] == num_dense_samples + + train_x_sparse.shape[0]

        print("Sampled train_x shape:", train_x_s.shape)
        print("Sampled train_y shape:", train_y_s.shape)

        self.scaler = StandardScaler().fit(train_x_s)

        return train_x_s, train_y_s

    def predict(self, test_x):
        """
            Predict labels for test data
        """

        y = self.model.predict(test_x)
        # we add a safety increase to 'safe' predictions since false negatives are penalized harshly
        predict_safe = (y < THRESHOLD).astype(int)
        y += 0.15 * predict_safe

        return y

    def fit_model(self, train_x, train_y):
        """
            Fit a Gaussian process regressor with noisy Matern kernel to the given data
        """

        train_x, train_y = self.preprocess(train_x, train_y, 1500)

        k = ker.Matern(length_scale=0.01, nu=2.5) + \
            ker.WhiteKernel(noise_level=1e-05)

        gpr = gp.GaussianProcessRegressor(kernel=k, alpha=0.01, n_restarts_optimizer=20, random_state=42, normalize_y=True)
        noisyMat_gpr = pipeline.Pipeline([
            ("scaler", self.scaler),
            ("gpr", gpr)
        ])


        print("Fitting noisy Matern GPR")
        start = time()
        noisyMat_gpr.fit(train_x, train_y)
        print("Took {} seconds".format(time() - start))

        self.model = noisyMat_gpr


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)


if __name__ == "__main__":
    main()
