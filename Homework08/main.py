import get_data
from model import MyCNN, MyAutoencoder, MyDecoder
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import numpy as np

config = "run2"
epochs = 15 # around 10
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

encoder = MyCNN(optimizer,embedding,filter_start = 24, regularizer=tf.keras.regularizers.L2(0.001))
decoder = MyDecoder(24)

model = MyAutoencoder(encoder,decoder)

# compile and fit
model.compile(optimizer = optimizer, loss=loss)

logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{config}")

history = model.fit(training_data, validation_data = val_data, epochs=epochs, batch_size=batch_size, callbacks=[logging_callback])

model.save(f"saved_model/{config}")
encoder.save(f"encoder/{config}")
decoder.save(f"decoder/{config}")

# latent space analysis
test_samples = get_data.data_preprocess(o_val_data.take(1000), batch_size = 1000, noisy = noise_std, targets = True)
for sample in test_samples:
    n,o,t = sample
    sample_embedding = encoder(n) # put noisy images in encoder to get the embeddings
    break

sample_embedding_reduced = TSNE(n_components=2).fit_transform(sample_embedding)

os.makedirs(f"Plots/", exist_ok = True)

plt.scatter(sample_embedding_reduced[:,0],sample_embedding_reduced[:,1],c=t)
plt.title("Embedding")
plt.savefig(f"Plots/{config}_embedding.png")
plt.show()

# Interpolation
how_many = 5
embedding1 = sample_embedding[0]
embedding2 = sample_embedding[1]
interpolation_embeddings = tf.linspace(embedding1,embedding2,how_many,axis=0)
interpolation_results = decoder(interpolation_embeddings)

f, axes = plt.subplots(1,how_many)
for i in range(how_many):
    axes[i].imshow(interpolation_results[i])
    # the titels are too long, only works if you make the plot full screen
    #axes[i].set_title(np.around(interpolation_embeddings[i].numpy(),decimals=2))
plt.savefig(f"Plots/{config}_interpolation.png")
plt.show()

# plot the training results
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(labels=["training","validation"])
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.savefig(f"Plots/{config}.png")
plt.show()