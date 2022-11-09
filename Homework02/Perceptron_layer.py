# implements a layer of perceptrons to be used in the MLP class

import numpy as np
import utils

class MLP_Layer:
    """ A Layer for creating an multi layer perceptron

    Attributes:
        biases = the biases of the units
        weights = the weights of the units
        layer_input = the last input of the layer
        layer_preactivation = the last calculated preactivation of the layer
        layer_activation = the last activation of the layer
        activation = the activation function of the layer
        activation_deriv = the derivative of the activation function
        learning_rate = the learning rate
    
    Methods: 
        forward_step: calculates the activation of each unit
        backward_step: updates the weights of the layer
    """

    def __init__(self,n_units : int, input_units : int,learning_rate = 0.02,activation = utils.ReLu,
                activation_deriv = utils.derivReLu):
        """ Constructor
        
        Parameters: 
            n_units (int) = the number of units the layer should have
            input_units (int) = the number of input units
            learning_rate (float) = the learning rate
            activation = the activation function
            activation_deriv = the derivative of the activation function
        """

        self.biases = np.zeros(shape=(n_units))
        self.weights = np.random.normal(size=(input_units,n_units))
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.learning_rate = learning_rate

    def forward_step(self,inputs):
        """ returns each unit's activation
        
        Parameters: 
            inputs (array) = the input given to the layer 
        """

        self.layer_input = inputs 
        self.layer_preactivation = ( inputs @ self.weights ) + self.biases
        self.layer_activation = self.activation(self.layer_preactivation)
        return self.layer_activation

    def backward_step(self,error):
        """ updates each unit's parameters using backpropagation given an error 

        Parameters: 
            error (array) = the derivative of the loss for the output layer, otherwise the next_error calculated by the layer l+1 
        """

        delta = self.activation_deriv(self.layer_preactivation) * error
        weight_change = self.layer_input.T @ delta

        next_error = delta @ self.weights.T

        self.weights -= self.learning_rate * weight_change
        self.biases -= ( self.learning_rate * delta[0] )

        return next_error