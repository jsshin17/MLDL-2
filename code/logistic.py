"""
A starting code for a logistic regression model.  This implementation should be based on the
minimum classification error heuristic.
"""

from numpy import *

from binary import *
import util
from regression import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Logistic(BinaryClassifier):
    """
    This class is for the logistic regression model implementation.
    """

    w = None
    Lambda = 0.0

    def __init__(self, opts):
        """
        Initialize our internal state.
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

        X= np.array(X)

        y_pred = np.dot(X, self.w) + self.b
        sum = 0
        for i in range(len(y_pred)):
            sum+=y_pred[i]- int(y_pred[i])
        return y_pred, 1/sum

    def train(self, X, Y):
        """
        Build a logistic regression model.
        """
        X = np.array(X)
        Y = np.array(Y)
        x = tf.placeholder(tf.float32, [None, 17])
        y = tf.placeholder(tf.float32, [None, 1])
        w = tf.Variable(tf.random_normal([17, 1]), name='weight')
        b = tf.Variable(tf.random_normal([3]), name='bias')

        hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(10001):
                sess.run(optimizer, feed_dict={x : X, y : Y})

            self.w, self.b = sess.run([w, b], feed_dict = {x : X, y : Y})
