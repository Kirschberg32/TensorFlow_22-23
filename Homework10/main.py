import tensorflow as tf

import data_processing
from model import getModel



# loading data
file_name = "bible.txt"
file_path = f"data/{file_name}"
data = data_processing.load_data(file_path)

#preprocesse
tokens = data_processing.nlp_preprocess(data,10000)

#Pairen
pairs = data_processing.pairing(tokens, 5)

# split in train and test set
pairs_len = len(pairs)
train_len = int(pairs_len * 0.8)
train_pairs, test_pairs = pairs[:train_len], pairs[train_len:]

#create datasets and preprocess
train_ds = tf.data.Dataset.from_tensor_slices(train_pairs)
test_ds = tf.data.Dataset.from_tensor_slices(test_pairs)

train_ds = data_processing.data_preprocess(train_ds)
test_data = data_processing.data_preprocess(test_ds)

model = getModel(10000,64,1)
model.compile(optimizer='adam', loss = tf.nn.nce_loss)
