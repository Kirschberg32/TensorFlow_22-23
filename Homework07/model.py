import tensorflow as tf

class RNNCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, recurrent_units_1, recurrent_units_2, **kwargs):
        super().__init__(**kwargs)

        self.recurrent_units_1 = recurrent_units_1
        self.recurrent_units_2 = recurrent_units_2
        
        self.linear_1 = tf.keras.layers.Dense(recurrent_units_1)
        self.linear_2 = tf.keras.layers.Dense(recurrent_units_2)
        
        # first recurrent layer in the RNN
        self.recurrent_layer_1 = tf.keras.layers.Dense(recurrent_units_1, 
                                                       kernel_initializer= tf.keras.initializers.Orthogonal(
                                                           gain=1.0, seed=None),
                                                       activation=tf.nn.tanh)
        # layer normalization for trainability
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        
        # second recurrent layer in the RNN
        self.recurrent_layer_2 = tf.keras.layers.Dense(recurrent_units_2, 
                                                       kernel_initializer= tf.keras.initializers.Orthogonal(
                                                           gain=1.0, seed=None), 
                                                       activation=tf.nn.tanh)
        # layer normalization for trainability
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
    
    @property
    def state_size(self):
        return [tf.TensorShape([self.recurrent_units_1]), 
                tf.TensorShape([self.recurrent_units_2])]
    @property
    def output_size(self):
        return [tf.TensorShape([self.recurrent_units_2])]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([self.recurrent_units_1]), 
                tf.zeros([self.recurrent_units_2])]

    def call(self, inputs, states):
        # unpack the states
        state_layer_1 = states[0]
        state_layer_2 = states[1]
        
        # linearly project input
        x = self.linear_1(inputs) + state_layer_1
        
        # apply first recurrent kernel
        new_state_layer_1 = self.recurrent_layer_1(x)
        
        # apply layer norm
        x = self.layer_norm_1(new_state_layer_1)
        
        # linearly project output of layer norm
        x = self.linear_2(x) + state_layer_2
        
        # apply second recurrent layer
        new_state_layer_2 = self.recurrent_layer_2(x)
        
        # apply second layer's layer norm
        x = self.layer_norm_2(new_state_layer_2)
        
        # return output and the list of new states of the layers
        return x, [new_state_layer_1, new_state_layer_2]
    
    def get_config(self):
        return {"recurrent_units_1": self.recurrent_units_1, 
                "recurrent_units_2": self.recurrent_units_2}
    
"""    
class MinimalRNNCell(AbstractRNNCell):

    def __init__(self, units, **kwargs):
      self.units = units
      super(MinimalRNNCell, self).__init__(**kwargs)

    @property
    def state_size(self):
      return self.units

    def build(self, input_shape):
      self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='uniform',
                                    name='kernel')
      self.recurrent_kernel = self.add_weight(
          shape=(self.units, self.units),
          initializer='uniform',
          name='recurrent_kernel')
      self.built = True

    def call(self, inputs, states):
      prev_output = states[0]
      h = backend.dot(inputs, self.kernel)
      output = h + backend.dot(prev_output, self.recurrent_kernel)
      return output, output
"""    
class MyRNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.rnn_cell = RNNCell(recurrent_units_1=24,
                                recurrent_units_2=48)
        
        # return_sequences collects and returns the output of the rnn_cell for all time-steps
        # unroll unrolls the network for speed (at the cost of memory)
        self.rnn_layer = tf.keras.layers.RNN(self.rnn_cell, return_sequences=False, unroll=True)
        
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        
        self.metrics_list = [tf.keras.metrics.Mean(name="loss"),
                             tf.keras.metrics.BinaryAccuracy()]
    
    @property
    def metrics(self):
        return self.metrics_list
    
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()
        
    def call(self, sequence, training=False):
        
        rnn_output = self.rnn_layer(sequence)
        
        return self.output_layer(rnn_output)
    
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