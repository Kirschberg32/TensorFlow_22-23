import data_processing as dp
import tensorflow as tf
from model import MyModel

# variables from prepare, do not change, or otherwise prepare newly
original_file_path = "data/bible.txt"
prepared_file_path = "data/prepared_bible.txt"
model_prefix = 'tokenizer_model'
VOCABULARY_SIZE = 2000 # 2000 - 7000
WINDOW_SIZE = 64 # 32 - 256
BATCH_SIZE = 64
EMBEDDING_DIM = 100 # 64 - 256
NUM_HEADS = 2 # 2-4
FIRST_UNIT = 32 # 32-256

# variables for the model
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = True)

# prepare if you want to create a new tokenizer and a new prepared data file
data, prepared_data, tokenizer = dp.prepare_everything(original_file_path,prepared_file_path,model_prefix,VOCABULARY_SIZE)

# if you only want to create a new tokenizer first, use loading afterwards
# dp.train_tokenizer(prepared_file_path,VOCABULARY_SIZE,model_prefix)

# load everything if already prepared
# data, prepared_data, tokenizer = dp.prepare_everything(original_file_path,prepared_file_path,model_prefix)

dataset = dp.create_dataset(prepared_data,tokenizer,WINDOW_SIZE, BATCH_SIZE)

# model = MyModel(tokenizer, optimizer, loss_function, VOCABULARY_SIZE, EMBEDDING_DIM, NUM_HEADS, FIRST_UNIT)