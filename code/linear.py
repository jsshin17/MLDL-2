"""
A starting code for a vanilla linear regression model.  This implementation should be based on the
minimum classification error heuristic.
"""

from binary import *
import util
from regression import *
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np

class Linear(Regression):
    """
    This class is for the decision tree implementation.  
    It has a partial implementation for the tree data structure. 
    This class also has a function to print the tree in a canonical form.
    """

    w = None

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """
        self.opts = opts

    def online(self):
        ### TODO: YOU MAY MODIFY THIS
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return self.w

    def __str__(self):
        """
        Return a string representation of the tree
        """
        return self.w

    def predict(self, X):
        """
        Perform inference
        """
        X = np.array(X)
        Y_predict = np.dot(X, self.w)
        return Y_predict

    def train(self, X, Y):
        """
        Build a linear regressor.
        """
        X = np.array(X)
        Y = np.array(Y)
        X_transpose = X.T
        temp_1 = np.dot(X_transpose, X)
        temp_2 = np.dot(inv(temp_1), X_transpose)
        self.w = np.dot(temp_2, Y)