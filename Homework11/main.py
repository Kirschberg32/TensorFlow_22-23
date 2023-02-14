import data_processing as dp
import tensorflow as tf
from model import MyModel
from training_loop import training_loop

# variables from prepare, do not change, or otherwise prepare newly
original_file_path = "data/bible.txt"
prepared_file_path = "data/prepared_bible.txt"
model_prefix = 'tokenizer_model'
VOCABULARY_SIZE = 2000 # 2000 - 7000
WINDOW_SIZE = 32 # 32 - 256
BATCH_SIZE = 64
EMBEDDING_DIM = 100 # 64 - 256
NUM_HEADS = 2 # 2-4
FIRST_UNIT = 32 # 32-256

starting_prompt = "What is "
EPOCHS_start = 0 # only needed if you want to continue training
EPOCHS_end = 1 # 100 - 600
TEST_OUTPUT_LENGTH = 10
TOP_K = 5

# Define where to save the log and model
config_name= "test1"
model_filepath = f'model/{config_name}'
train_log_path = f"logs/{config_name}/train"
train_summary_writer = tf.summary.create_file_writer(train_log_path)

# variables for the model
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = True)

# Data creation
#**************

# prepare if you want to create a new tokenizer and a new prepared data file
# data, prepared_data, tokenizer = dp.prepare_everything(original_file_path,prepared_file_path,model_prefix,VOCABULARY_SIZE)

# if you only want to create a new tokenizer first, use loading afterwards
# dp.train_tokenizer(prepared_file_path,VOCABULARY_SIZE,model_prefix)

# load everything if already prepared
data, prepared_data, tokenizer = dp.load_everything(original_file_path,prepared_file_path,model_prefix)

dataset = dp.create_dataset(prepared_data,tokenizer,WINDOW_SIZE, BATCH_SIZE)

# Model and training
#*******************

model = MyModel(tokenizer, optimizer, loss_function, VOCABULARY_SIZE, WINDOW_SIZE, EMBEDDING_DIM, NUM_HEADS, FIRST_UNIT)
# model.load_weights(model_filepath)

training_loop(model,dataset,EPOCHS_start, EPOCHS_end,starting_prompt, TEST_OUTPUT_LENGTH, TOP_K, train_summary_writer,model_filepath)