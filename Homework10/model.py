import tensorflow as tf 

class SkipGram(tf.keras.models.Model):
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

    def call(self, input, training = None): # not done
        embedding = tf.nn.embedding_lookup(self.embedding, input)
        output = tf.reduce_sum(embedding, axis=1)
        output = tf.nn.softmax(output)
        return output

    def train_step(self,inputs):
        pass