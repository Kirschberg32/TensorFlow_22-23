import tensorflow as tf

from data_processing import get_preprocessed_data
#from model import getModel

# loading data
file_name = "bible.txt"
file_path = f"data/{file_name}"

VOCABULARY = 10000
WINDOW = 2 # because window is i - 2 and i + 2 then = 5
TRAIN_PART = 0.8
EMBEDDING = 64
print("Start funktion")
(train_ds, test_ds), tokenizer = get_preprocessed_data(file_path,VOCABULARY,WINDOW,TRAIN_PART)
print("Back in main")
#model = getModel(VOCABULARY,EMBEDDING,1)
#model.compile(optimizer='adam', loss = tf.nn.nce_loss)

for element in train_ds:
    print(element)
    break