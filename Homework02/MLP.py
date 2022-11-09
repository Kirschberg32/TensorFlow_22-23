from Perceptron_layer import MLP_Layer
import utils

class MLP:
    """ A multi layered Perceptron 

    Attributes: 
        hidden_layers (list) = a list of the hidden layers of the mlp
        output_layer (MLP_Layer) = the last (output) layer of the mlp having a linear activation function
    
    Methods: 
        forward_step: feeds input to the network and returns the networks output
        backpropagation: updates all the networks parameters using backpropagation
    """

    def __init__(self,hidden_layers : list,input_units : int, output_units : int):
        """ Constructor

        Parameters:
            hidden_layers (list)= list with the length giving the number of hidden layers and for each layer the elements value gives the number of units 
                (e.g. [1,2,3,4] creates a network with 4 hidden layers which have 1,2,3 and lastly 4 units)
            input_units (int) = the length of the input array
            output_units (int) = the number of units in the output layer
         """

        self.hidden_layers = [MLP_Layer(hu,input_units) if i == 0 else MLP_Layer(hu,hidden_layers[i-1]) for i,hu in enumerate(hidden_layers)]
        self.output_layer = MLP_Layer(output_units,hidden_layers[-1],activation=utils.linear,activation_deriv=utils.derivLinear)

    def forward_step(self,inputs):
        """ passes an input through the entire network 
        
        Parameters:
            inputs (array) (input should have 2 dimensions) 
        """

        x = inputs
        for hl in self.hidden_layers:
            x = hl.forward_step(x)
        return self.output_layer.forward_step(x)

    def backpropagation(self,loss):
        """ updates all the weights and biases in the network
        
        Paramters:
            loss (array) = 2 dimensional array containing the derivative of the loss
        """

        loss = self.output_layer.backward_step(loss)

        for hl in reversed(self.hidden_layers):
            loss = hl.backward_step(loss)
