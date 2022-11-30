import tensorflow as tf

class MyCNNBlock(tf.keras.layers.Layer):
    """ a block for a CNN having several convoluted layers with filters and kernel size 3 and ReLu as the activation function """

    def __init__(self,layers,filters,global_pool = False):
        """ Constructor 
        
        Parameters: 
            layers (int) = how many Conv2D you want
            filters (int) = how many filters the Conv2D layers should have
            global_pooling (boolean) = global average pooling at the end if True else MaxPooling2D
        """

        super(MyCNNBlock, self).__init__()
        self.conv_layers =  [tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu') for _ in range(layers)]
        self.pool = self.global_pool = tf.keras.layers.GlobalAvgPool2D() if global_pool else tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

    @tf.function
    def call(self,x):
        """ forward propagation of this block """
        for layer in self.conv_layers:
            x = layer(x)
        x = self.pool(x)
        return x

class MyCNN(tf.keras.Model):
    """ an CNN created to train on Cifar-10 """
    
    def __init__(self, optimizer,output_units : int = 10):
        """ Constructor 
        
        Parameters: 
            optimizer = the optimizer to use for training
            output_units (int) = the number of wanted output units
        """

        super(MyCNN, self).__init__()

        # architecture
        self.block1 = MyCNNBlock(layers = 2,filters = 24)
        self.block2 = MyCNNBlock(layers = 2,filters = 48)
        self.block3 = MyCNNBlock(layers = 2,filters = 96,global_pool=True)

        self.dense1 = tf.keras.layers.Dense(128,activation = tf.nn.relu)
        self.out = tf.keras.layers.Dense(output_units, activation=tf.nn.softmax)

        self.loss_function = tf.losses.CategoricalCrossentropy()
        self.optimizer = optimizer

        self.metrices = [tf.keras.metrics.Mean(name = "loss"), tf.keras.metrics.CategoricalAccuracy(name = "accuracy")]

    def reset_metrics(self):
        """ resets all the metrices that are observed during training and testing """
        for m in self.metrics:
            m.reset_states()

    @tf.function
    def call(self, x, training = False):
        """ forward propagation of the ANN """

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.dense1(x)
        x = self.out(x)
        return x

    @tf.function
    def train_step(self,data):
        """ does one train step in one episode given a batch of data 
        
        Parameters: 
            data (tuple) = shape (img, target)
        
        returns a dictionary of the metrics
        """

        img, targets = data

        with tf.GradientTape() as tape: 

            predictions = self(img,training=True)
            loss = self.loss_function(targets, tf.squeeze(predictions))

            self.metrics[0].update_state(values = loss) # loss
            self.metrics[1].update_state(predictions, targets) # accuracy

        # get the gradients
        gradients = tape.gradient(loss,self.trainable_variables)

        # apply the gradient
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return{m.name : m.result() for m in self.metrics}

    @tf.function
    def test_step(self,data):
        """ does one test step in one episode given a batch of data 
        
        Parameters: 
            data (tuple) = shape (img, target)
        
        returns a dictionary of the metrics
        """

        img, targets = data

        predictions = self(img,training=False)
        loss = self.loss_function(targets, tf.squeeze(predictions))

        self.metrics[0].update_state(values = loss) # loss
        self.metrics[1].update_state(predictions, targets) # accuracy

        return{m.name : m.result() for m in self.metrics}