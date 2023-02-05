import tensorflow as tf 

class SkipGram(tf.keras.models.Model):
    """ A SkipGram model to create word embeddings. """
    
    def __init__(self, vocabulary_size : int, embedding_size : int = 64, num_neg_sample : int = 64):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_neg_sample = num_neg_sample

        self.metrices = [tf.keras.metrics.Mean(name = "loss")]

    #def build(self, input_shape):
        #self.num_sampled = input_shape[2]
        self.embedding = self.add_weight(
            name = "embedding",
            shape=(self.vocabulary_size,self.embedding_size),
            initializer = "uniform",
            trainable = True
        )
        self.score_weight = self.add_weight(
            name='score',
            shape=(self.vocabulary_size,self.embedding_size),
            initializer='uniform',
            trainable=True
        )
        self.score_bias = self.add_weight(
            name='score_bias',
            shape=(self.vocabulary_size, ),
            initializer='zero',
            trainable=True
        )

    def reset_metrics(self):
        """ resets all the metrices that are observed during training and testing """
        for m in self.metrics:
            m.reset_states()

    @tf.function
    def call(self, input, training = None): # not done
        target_embedding = tf.nn.embedding_lookup(self.embedding, input[:][0])
        loss = tf.nn.nce_loss(
                weights=self.score_weight,#(100,embedding_size)
                biases=self.score_bias, #(100,)
                labels=input[:][1], #(batch-size,1)
                inputs=target_embedding, # (batch_size,embedding_size)
                num_true = 1, #self.num_sampled, # jetzt 1
                num_sampled = self.num_neg_sample, # 1
                num_classes = self.vocabulary_size # 100
            )
        return loss

    @tf.function
    def train_step(self,inputs):

        with tf.GradientTape() as tape:
            loss = self(inputs,True)
            self.metrics[0].update_state(values = tf.mean(loss,reduce_instance_dims = False)) # loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        
        return {m.name : m.result() for m in self.metrics}

