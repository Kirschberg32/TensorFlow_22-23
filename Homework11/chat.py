import data_processing as dp
import tensorflow as tf
from model import MyModel

# variables from prepare, do not change, or otherwise prepare newly
original_file_path = "data/HP1.txt"
prepared_file_path = "data/prepared_HP1.txt"
model_prefix = 'tokenizer_model'
VOCABULARY_SIZE = 2000 # 2000 - 7000
WINDOW_SIZE = 32 # 32 - 256
BATCH_SIZE = 64
EMBEDDING_DIM = 100 # 64 - 256
NUM_HEADS = 2 # 2-4
FIRST_UNIT = 32 # 32-256

TOP_K = 20

model_to_load_path = f'model/run1_HP_10epochs/0'


# Model
#******

# variables for the model
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = True)

# load everything if already prepared
data, prepared_data, tokenizer = dp.load_everything(original_file_path,prepared_file_path,model_prefix)

model = MyModel(tokenizer, optimizer, loss_function, VOCABULARY_SIZE, WINDOW_SIZE, EMBEDDING_DIM, NUM_HEADS, FIRST_UNIT)
model.generate_text("hello dear,", 1, 2)
model.load_weights(model_to_load_path)

print("Chat is now ready for your first input.")
print("Exit by typing exit.")

for i in range(10000):
    text_in = input().lower()
    if text_in == "exit":
        print("Chat is now done.")
        break
    output = model.generate_text(text_in,20,TOP_K)
    print("Chat answers: ", output.numpy())
