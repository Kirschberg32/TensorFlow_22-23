import tensorflow as tf 

class SkipGram(tf.keras.layers.Layer):
    """ A SkipGram model to create word embeddings. """
    
    def __init__(self, vocabulary_size, embedding_size : int = 64):

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name = "embedding",
            shape=(self.vocabulary_size,self.embedding_size),
            initializer = "uniform",
            trainiable = True
        )
        self.score = self.add_weight(
            name='score',
            shape=(self.vocabulary_size,self.embedding_size),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training = None): # not done
        output = tf.nn.embedding_lookup(self.embedding, inputs)
        output = tf.reduce_sum(output, axis=1)
        output = tf.nn.softmax(output)
        return output

def getModel(vocabulary_size : int = 10000, embedding_size : int = 64, input_shape : int = 1):
    """ creates a skipgram model """

    inputs = tf.keras.layers.Input(shape = (input_shape, ), dtype='int32')
    outputs = SkipGram(vocabulary_size,embedding_size)(inputs)

    model = tf.keras.models.Model(inputs = inputs,outputs = outputs)
    return model