import tensorflow as tf

class MyModel(tf.keras.Model):
    """ an ANN created to train on the mnist dataset """
    
    def __init__(self,hidden_units : list,output_units : int = 10,hidden_activation = tf.nn.relu):
        """ Constructor 
        
        Parameters: 
            hidden_units (list) = list containing one element for each hidden layer and the values are the units of each layer 
            output_units (int) = the number of wanted output units
            hidden_activation = the activation function for the hidden layers
        """

        super(MyModel, self).__init__()
        self.dense_list = [ tf.keras.layers.Dense(units, activation=hidden_activation) for units in hidden_units ]
        self.out = tf.keras.layers.Dense(output_units, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        """ forward propagation of the ANN """
        x = inputs
        for layer in self.dense_list:
            x = layer(x)
        x = self.out(x)
        return x