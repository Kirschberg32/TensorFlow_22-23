import tensorflow as tf

class MyModel(tf.keras.Model):
    """ an ANN created to train on the double mnist dataset (Math)
    it is a shared weight model 
    """
    
    def __init__(self,loss_function, optimizer, hidden_units : list,output_units : int = 1,hidden_activation = tf.nn.relu,output_activation = tf.nn.softmax):
        """ Constructor 
        
        Parameters: 
            loss_function = the loss function to use for training
            optimizer = the optimizer to use for training
            hidden_units (list) = list containing one element for each hidden layer and the values are the units of each layer (should have at least a length of 1)
            output_units (int) = the number of wanted output units
            hidden_activation = the activation function for the hidden layers
            output_activation = the activation function to use for the output layer
        """

        super(MyModel, self).__init__()
        self.first_layer = tf.keras.layers.Dense(hidden_units[0],activation=hidden_activation)
        self.concat_layer = tf.keras.layers.Concatenate()
        self.dense_list = [ tf.keras.layers.Dense(units, activation=hidden_activation) for i,units in enumerate(hidden_units) if i != 0]
        self.out = tf.keras.layers.Dense(output_units, activation=output_activation)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrices = [tf.keras.metrics.Mean(name = "loss")]

    def reset_metrics(self):
        """ resets all the metrices that are observed during training and testing """
        for m in self.metrics:
            m.reset_states()

    @tf.function
    def call(self, input1, input2, training = False):
        """ forward propagation of the ANN """

        x = self.first_layer(input1)
        y = self.first_layer(input2)

        # concatenate the two results
        x = self.concat_layer([x, y])

        for layer in self.dense_list:
            x = layer(x)
        x = self.out(x)
        return x

    @tf.function
    def train_step(self,data):
        """ does one train step in one episode given a batch of data 
        
        Parameters: 
            data (tuple) = shape (img1,img2, target)
        
        returns a dictionary of the metrics
        """

        (img1, img2, targets) = data

        with tf.GradientTape() as tape: 

            predictions = self(img1,img2,training=True)
            loss = self.loss_function(targets, tf.squeeze(predictions))

            self.metrics[0].update_state(values = loss) # loss

        # get the gradients
        gradients = tape.gradient(loss,self.trainable_variables)

        # apply the gradient
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return{m.name : m.result() for m in self.metrics}

    @tf.function
    def test_step(self,data):
        """ does one test step in one episode given a batch of data 
        
        Parameters: 
            data (tuple) = shape (img1,img2, target)
        
        returns a dictionary of the metrics
        """

        (img1, img2, targets) = data

        predictions = self(img1,img2,training=False)
        loss = self.loss_function(targets, tf.squeeze(predictions))

        self.metrics[0].update_state(values = loss) # loss

        return{m.name : m.result() for m in self.metrics}