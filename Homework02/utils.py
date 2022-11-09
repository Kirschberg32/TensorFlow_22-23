# implements important functions for the MLP

import numpy as np

def ReLu(x):
    """ calculates ReLu for a value or an array """

    return np.where(x>=0,x,0)

def derivReLu(x):
    """ calculates the derivative of ReLu for a value or an array """

    return np.where(x>=0,1,0)

def Loss(y,t):
    """ caluclates the Loss, which is the mean squared error given the values (y) and targets (t)"""

    return 1/2 * (y - t)**2

def derivLoss(y,t):
    """ calculates the derivative of the Loss given values (y) and targets (t) """

    return y - t

def linear(x):
    """ a linear activation function """
    return x

def derivLinear(x):
    """ the derivative of a linear activation function """
    return np.ones(shape = x.shape)