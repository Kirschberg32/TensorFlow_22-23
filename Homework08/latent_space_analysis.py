import get_data
from model import MyCNN, MyAutoencoder, MyDecoder
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import numpy as np

config_in = "run1"
config_out = "testtsne1"
batch_size = 64
noise_std = 0.2
embedding = 10

optimizer = tf.optimizers.Adam(learning_rate=0.001)
loss = tf.losses.MeanSquaredError()

# get and prepare data and model

(training_data, o_val_data ) , ds_info = get_data.load_data(False) # True
#training_data = training_data.take(10) # 
training_data = get_data.data_preprocess(training_data, batch_size = batch_size,noisy = noise_std)
val_data = get_data.data_preprocess(o_val_data, batch_size = batch_size, noisy = noise_std)

encoder = tf.keras.models.load_model(f"saved_encoder/{config_in}")
decoder = tf.keras.models.load_model(f"saved_decoder/{config_in}")

# latent space analysis

test_samples = get_data.data_preprocess(o_val_data, batch_size = 1000, noisy = noise_std, targets = True)
for sample in test_samples:
    n,o,t = sample
    sample_embedding = encoder(n) # put noisy images in encoder to get the embeddings
    break

print(tf.reduce_min(sample_embedding))
print(tf.reduce_max(sample_embedding))

sample_embedding_reduced = TSNE(n_components=2).fit_transform(sample_embedding)

os.makedirs(f"Plots/", exist_ok = True)

plt.scatter(sample_embedding_reduced[:,0],sample_embedding_reduced[:,1],c=t)
plt.title("Embedding")
plt.savefig(f"Plots/{config_out}_embedding.png")
plt.show()

# Interpolation

how_many = 5
embedding1 = sample_embedding[0]
embedding2 = sample_embedding[100]
interpolation_embeddings = tf.linspace(embedding1,embedding2,how_many,axis=0)
interpolation_results = decoder(interpolation_embeddings)

f, axes = plt.subplots(1,how_many)
for i in range(how_many):
    axes[i].imshow(interpolation_results[i])
    # the titels are too long, only works if you make the plot full screen
    #axes[i].set_title(np.around(interpolation_embeddings[i].numpy(),decimals=2))
plt.savefig(f"Plots/{config_out}_interpolation.png")
plt.show()