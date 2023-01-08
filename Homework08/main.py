import get_data
from model import MyCNN, MyAutoencoder, MyDecoder
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os

config = "testrun1"
epochs = 2 # around 10
batch_size = 64
noise_std = 0.2
embedding = 10

optimizer = tf.optimizers.Adam(learning_rate=0.001)
loss = tf.losses.MeanSquaredError()

# get and prepare data and model
(training_data, val_data ) , ds_info = get_data.load_data(False) # True
#training_data = training_data.take(2000) # 
training_data = get_data.data_preprocess(training_data, batch_size = batch_size,noisy = noise_std)
val_data = get_data.data_preprocess(val_data, batch_size = batch_size, noisy = noise_std)

encoder = MyCNN(optimizer,embedding,filter_start = 24, regularizer=tf.keras.regularizers.L2(0.001))
decoder = MyDecoder(24)

model = MyAutoencoder(encoder,decoder)

# compile and fit
model.compile(optimizer = optimizer, loss=loss)

logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{config}")

history = model.fit(training_data, validation_data = val_data, epochs=epochs, batch_size=batch_size, callbacks=[logging_callback])
os.makedirs(f"Plots/", exist_ok = True)

model.save(f"saved_model/{config}")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(labels=["training","validation"])
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.savefig(f"Plots/{config}.png")
plt.show()