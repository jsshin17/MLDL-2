"""
A starting code for a vanilla linear regression model.  This implementation should be based on the
minimum classification error heuristic.
"""

from numpy import *

from binary import *
import util
from regression import *
import numpy as np
from numpy.linalg import inv

class Ridge(Regression):
    """
    This class is for the ridge regressor implementation.
    """

    w = None
    Lambda = 0.0

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

    def setLambda(self, lambdaVal):
        self.Lambda = lambdaVal

    def online(self):
        ### TODO: YOU MAY MODIFY THIS
        return False

    def __repr__(self):
        """
        Return a string representation of the model
        """
        return self.w

    def __str__(self):
        """
        Return a string representation of the model
        """
        return self.w

    def predict(self, X):
        """
        Perform inference
        """
        X = np.array(X)
        prediction = np.dot(X, self.w)
        return prediction

    def train(self, X, Y):
        """
        Build a ridge regressor.
        """
        X = np.array(X)
        Y = np.array(Y)
        X_transpose = X.T
        temp_1 = np.dot(X_transpose, X)
        temp_2 = np.dot(X_transpose, Y)
        identity = np.eye(X.shape[1])
        identity[0, 0] = 0
        regularizer = self.Lambda * identity
        temp_1 = temp_1 + regularizer
        self.w = np.dot(inv(temp_1), temp_2)


