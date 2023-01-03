import tensorflow as tf

class MyLSTMCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)

        self.units = units

        self.inputConcat = tf.keras.layers.Concatenate(axis=-1)

        # layer for forget gate using sigmoid
        self.sigmoid1_layer = tf.keras.layers.Dense(self.units, 
                                                kernel_initializer=tf.keras.initializers.Orthogonal(
                                                    gain=1.0, seed=None), 
                                                activation=tf.nn.sigmoid)

        self.multiplier = tf.keras.layers.Multiply()

        # layer for input gate using sigmoid
        self.sigmoid2_layer = tf.keras.layers.Dense(self.units, 
                                                kernel_initializer=tf.keras.initializers.Orthogonal(
                                                    gain=1.0, seed=None), 
                                                activation=tf.nn.sigmoid)
        
        # layer for input gate using tanh
        self.tanh_layer = tf.keras.layers.Dense(self.units, 
                                                kernel_initializer=tf.keras.initializers.Orthogonal(
                                                    gain=1.0, seed=None), 
                                                activation= tf.nn.tanh)
        self.adder = tf.keras.layers.Add()

        # layer for output gate using sigmoid
        self.sigmoid3_layer = tf.keras.layers.Dense(self.units, 
                                                kernel_initializer=tf.keras.initializers.Orthogonal(
                                                    gain=1.0, seed=None), 
                                                activation=tf.nn.sigmoid)

        self.tanh = tf.keras.layers.Activation("tanh")
        
        # layer normalization for trainability
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
    
    @property
    def state_size(self):
        return [tf.TensorShape([self.units]), 
                tf.TensorShape([self.units])]
    @property
    def output_size(self):
        return [tf.TensorShape([self.units])]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # two entries of list, one beeing the hidden and the other the cell state
        # each entry has to contain the batch_size too
        return [tf.zeros([batch_size,self.units]), tf.zeros([batch_size,self.units])]

    def call(self, inputs, states):
        # unpack the states
        hidden_state = states[0]
        cell_state = states[1]

        # concat both as input
        x = self.inputConcat([inputs,hidden_state])
        x_forget = self.sigmoid1_layer(x)
        new_cell_state = self.multiplier([cell_state,x_forget])
        x_input1 = self.sigmoid2_layer(x)
        x_input2 = self.tanh_layer(x)
        x_input = self.multiplier([x_input1,x_input2])
        new_cell_state = self.adder([new_cell_state,x_input])
        x_output = self.sigmoid3_layer(x)
        new_hidden_state = output = self.multiplier([self.tanh(new_cell_state),x_output])
        
        return output, [new_hidden_state, new_cell_state]
    
    def get_config(self):
        return {"hidden_units": self.units}
    
class MyCNNBlock(tf.keras.layers.Layer):
    """ a block for a CNN having several convoluted layers with filters and kernel size 3 and ReLu as the activation function """

    def __init__(self,layers,filters,input_shape = None,global_pool = False,mode = False, reg = None, dropout_layer = None):
        """ Constructor 
        
        Parameters: 
            layers (int) = how many Conv2D you want
            filters (int) = how many filters the Conv2D layers should have
            input_shape (tuple) = just the input that is not supposed to be batches (e.g. batches and sequences at the top)
            global_pool (boolean) = global average pooling at the end if True else MaxPooling2D, if None then no pooling at all
            mode (string) = whether we want to implement a denseNet ("dense") or a ResNet "res" or none of them (None)
            reg = the Regularizer to use
            dropout_layer = the dropout layer to use
        """

        super(MyCNNBlock, self).__init__()

        self.dropout_layer = dropout_layer
        self.conv_layers =  [tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', kernel_regularizer = reg, input_shape=input_shape) for _ in range(layers)]

        self.mode = mode
        switch_mode = {"dense":tf.keras.layers.Concatenate(axis=-1), "res": tf.keras.layers.Add(),}
        self.extra_layer = None if mode == None else switch_mode.get(mode,f"{mode} is not a valid mode for MyCNN. Choose from 'dense' or 'res'.")

        self.pool = global_pool
        if global_pool is not None:
            self.pool = tf.keras.layers.GlobalAvgPool2D() if global_pool else tf.keras.layers.MaxPooling2D(pool_size=2, strides=2,input_shape = input_shape)

    @tf.function
    def call(self,input,training=None):
        """ forward propagation of this block """
        x = input
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if(i==0 and self.mode == "res"): # for resnet add output of first layer to final output, not input of first layer
                input = x
            if self.dropout_layer:
                x = self.dropout_layer(x, training)
        if(self.extra_layer is not None):
            x = self.extra_layer([input,x])

        if(self.pool != None):
            x = self.pool(x)
        return x
 
class MyLSTMModel(tf.keras.Model):
    def __init__(self, total_input_shape, lstm_units = 12, output_units : int = 1, mode = None,dropout_rate = None, regularizer = None):
        super().__init__()

        self.reg = regularizer
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate) if self.dropout_rate else None
        self.loss_function = tf.losses.MeanSquaredError()
        
        self.conv_block1 = MyCNNBlock(layers = 2, filters = 24, input_shape= total_input_shape[2:], global_pool = None, mode = mode, reg = self.reg, dropout_layer = self.dropout_layer) # Input: (batch_size,sequence,28,28,1)

        self.global_pooling = tf.keras.layers.GlobalAvgPool2D()
        self.timedistributed = tf.keras.layers.TimeDistributed(self.global_pooling)
        
        self.lstm_cell = MyLSTMCell(units=lstm_units)
        
        # return_sequences collects and returns the output of the rnn_cell for all time-steps
        # unroll unrolls the network for speed (at the cost of memory)
        self.rnn_buffer =tf.keras.layers.RNN(self.lstm_cell, return_sequences=True)
        
        self.output_layer = tf.keras.layers.Dense(output_units, activation="linear")
        
        self.metrics_list = [tf.keras.metrics.Mean(name = "loss"),
                             tf.keras.metrics.Accuracy(name="accuracy")]
    
    @property
    def metrics(self):
        return self.metrics_list
    
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()
        
    @tf.function
    def call(self, x, training=False):
        
        x = self.conv_block1(x,training = training)

        x = self.timedistributed(x)
        x = self.rnn_buffer(x)
        
        return self.output_layer(x)
    
    def train_step(self, data):
        
        """
        Standard train_step method, assuming we use model.compile(optimizer, loss, ...)
        """
        
        sequence, label = data
        with tf.GradientTape() as tape:
            output = self(sequence, training=True)
            loss = self.compiled_loss(label, output, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(label, output)
        
        return {m.name : m.result() for m in self.metrics}
    
    def test_step(self, data):
        
        """
        Standard test_step method, assuming we use model.compile(optimizer, loss, ...)
        """
        
        sequence, label = data
        output = self(sequence, training=False)
        loss = self.compiled_loss(label, output, regularization_losses=self.losses)
                
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(label, output)
        
        return {m.name : m.result() for m in self.metrics}