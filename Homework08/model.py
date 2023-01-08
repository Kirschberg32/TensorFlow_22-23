import tensorflow as tf

class MyCNNNormalizationLayer(tf.keras.layers.Layer):
    """ a layer for a CNN with kernel size 3 and ReLu as the activation function """

    def __init__(self,filters,normalization=False, reg = None):
        """ Constructor
        
        Parameters: 
            filters (int) = how many filters the Conv2D layer will have
            normalization (boolean) = whether the output of the layer should be normalized 
        """
        super(MyCNNNormalizationLayer, self).__init__()
        self.conv_layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', kernel_regularizer = reg)
        self.norm_layer = tf.keras.layers.BatchNormalization() if normalization else None
        self.activation = tf.keras.layers.Activation("relu")

    @tf.function
    def call(self,x,training=None):
        """ forward propagation """

        x = self.conv_layer(x)
        if self.norm_layer:
            x = self.norm_layer(x,training)
        x = self.activation(x)

        return x

class MyCNNBlock(tf.keras.layers.Layer):
    """ a block for a CNN having several convoluted layers with filters and kernel size 3 and ReLu as the activation function """

    def __init__(self,layers,filters,global_pool = False,mode = False,normalization = False, reg = None, dropout_layer = None):
        """ Constructor 
        
        Parameters: 
            layers (int) = how many Conv2D you want
            filters (int) = how many filters the Conv2D layers should have
            global_pool (boolean) = global average pooling at the end if True else MaxPooling2D
            denseNet (boolean) = whether we want to implement a denseNet (creates a concatenate layer if True)
        """

        super(MyCNNBlock, self).__init__()
        self.dropout_layer = dropout_layer
        self.conv_layers =  [MyCNNNormalizationLayer(filters,normalization, reg) for _ in range(layers)]
        self.mode = mode
        switch_mode = {"dense":tf.keras.layers.Concatenate(axis=-1), "res": tf.keras.layers.Add(),}
        self.extra_layer = None if mode == None else switch_mode.get(mode,f"{mode} is not a valid mode for MyCNN. Choose from 'dense' or 'res'.")
        self.pool = tf.keras.layers.GlobalAvgPool2D() if global_pool else tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

    @tf.function
    def call(self,input,training=None):
        """ forward propagation of this block """
        x = input
        for i, layer in enumerate(self.conv_layers):
            x = layer(x,training)
            if(i==0 and self.mode == "res"): # for resnet add output of first layer to final output, not input of first layer
                input = x
            if self.dropout_layer:
                x = self.dropout_layer(x, training)
        if(self.extra_layer is not None):
            x = self.extra_layer([input,x])

        x = self.pool(x)
        return x

class MyCNN(tf.keras.Model):
    """ an CNN created to train on Cifar-10 """
    
    def __init__(self, optimizer,output_units : int = 10, mode = None,normalization = False,dropout_rate = None, regularizer = None):
        """ Constructor 
        
        Parameters: 
            optimizer = the optimizer to use for training
            output_units (int) = the number of wanted output units
            mode (String) = whether to implement a DenseNet "dense" ore a ResNet "res"
            normalization (boolean) = whether to have normalization layers
            dropout (0<= int <1) = rate of dropout for after input and after dense
            regularizer (0<= int <1) = rate for l1 and l2 regularizer
        """

        super(MyCNN, self).__init__()

        # architecture
        self.reg = regularizer
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate) if self.dropout_rate else None
        self.block1 = MyCNNBlock(layers = 2,filters = 24,mode = mode,normalization = normalization, reg = self.reg, dropout_layer = self.dropout_layer)
        self.block2 = MyCNNBlock(layers = 2,filters = 48,mode = mode,normalization = normalization, reg = self.reg, dropout_layer = self.dropout_layer)

        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(output_units, activation=tf.nn.softmax)

        self.loss_function = tf.losses.CategoricalCrossentropy()
        self.optimizer = optimizer

    @tf.function
    def call(self, x, training = False):
        """ forward propagation of the ANN """
        
        x = self.block1(x,training = training)
        x = self.block2(x,training = training)

        x = self.flatten(x)
        x = self.out(x,training = training)
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
            loss = self.loss_function(targets, predictions, regularization_losses=self.losses)

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
        loss = self.loss_function(targets, predictions, regularization_losses=self.losses)

        self.metrics[0].update_state(values = loss) # loss
        self.metrics[1].update_state(predictions, targets) # accuracy

        return{m.name : m.result() for m in self.metrics}

class MyAutoencoder(tf.keras.Model): 

    def __init__(self,encoder,decoder,):

        super(MyCNN, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.metrices = [tf.keras.metrics.Mean(name = "loss"), tf.keras.metrics.Accuracy(name = "accuracy")]

    def reset_metrics(self):
        """ resets all the metrices that are observed during training and testing """
        for m in self.metrics:
            m.reset_states()

    @tf.function
    def call(self, x, training = False):
        """ forward propagation of the Autoencoder"""
        
        x = self.encoder(x,training = training)
        x = self.decoder(x,training = training)

        return x

    @tf.function
    def train_step(self, data):
        
        """
        Standard train_step method, assuming we use model.compile(optimizer, loss, ...)
        """
        
        image, target = data
        with tf.GradientTape() as tape:
            output = self(image, training=True)
            loss = self.compiled_loss(target, output, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(target, output)
        
        return {m.name : m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        
        """
        Standard test_step method, assuming we use model.compile(optimizer, loss, ...)
        """
        
        image, target = data
        output = self(image, training=False)
        loss = self.compiled_loss(target, output, regularization_losses=self.losses)
                
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(target, output)
        
        return {m.name : m.result() for m in self.metrics}