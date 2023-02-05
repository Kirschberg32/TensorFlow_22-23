import tensorflow as tf
import tqdm
import datetime

from data_processing import get_preprocessed_data
from model import SkipGram

# loading data
file_name = "bible.txt"
file_path = f"data/{file_name}"

config_name = "test1"

VOCABULARY = 100 #10000
WINDOW = 1 # because window is i - 2 and i + 2 then = 5
TRAIN_PART = 0.8 # partition of the data to be training data
EMBEDDING = 32 # size of the embedding
BATCH = 64 # batch_size
EPOCHS = 1
K = 5
NEGATIVE_SAMPLES = 1

words_keep_track = ["holy", "father", "wine", "poison", "love",
"strong", "day"]

(train_ds, test_ds), tokenizer = get_preprocessed_data(file_path,VOCABULARY,WINDOW,TRAIN_PART,BATCH)
words_sequence = tokenizer.texts_to_sequences([words_keep_track])[0]
cosine_similarity = tf.keras.losses.CosineSimilarity(axis=1)

model = SkipGram(VOCABULARY,EMBEDDING,NEGATIVE_SAMPLES)
model.compile(optimizer='adam',loss = tf.losses.BinaryCrossentropy())

# build the model
for s in train_ds:
    #model(s,False)
    print("Embedding: ", tf.nn.embedding_lookup(model.embedding, s[:,0]))
    break

#training loop 

time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# custom instead of time_string
config = config_name

train_file_path = f"logs/{config_name}/{time_string}/train"
test_file_path = f"logs/{config_name}/{time_string}/test"
train_summary_writer = tf.summary.create_file_writer(train_file_path)
test_summary_writer = tf.summary.create_file_writer(test_file_path)

for e in range(EPOCHS):

    for s in tqdm.tqdm(train_ds,position=0,leave=True):
        metrics = model.train_step(s)
        break # TODO

    # log in tensorboard and print
    with train_summary_writer.as_default():
        [tf.summary.scalar(name = m.name, data = m.result(),step=e) for m in model.metrics]

    [ tf.print(f"Epoch {e} {k}: {v.numpy()} ") for (k,v) in metrics.items() ]

    model.reset_metrics()

    print("\nEpoch: ", e)
    print("Evaluation k-nearest neighbours using cosine similarity")

    # calculate whole embedding 
    whole_embedding = [tf.nn.embedding_lookup(model.embedding, i) for i in range(VOCABULARY)]

    # calculate embedding of words
    track_words_embedding = [tf.nn.embedding_lookup(model.embedding, w) for w in words_sequence]

    for tw,j in enumerate(track_words_embedding):
        # calculate cosine similarities between whole and words 
        cosines = [(cosine_similarity(tw,we),i) for we,i in enumerate(whole_embedding)]

        # sort by distance and return k-nearest
        sorted_cosines = sorted(cosines)

        # sequence to text of nearest neighbours
        words_neighbors = tokenizer.sequences_to_texts([sorted_cosines[:K][1]])[0]

        # print word with its k-nearest (maybe with cosine similarities)
        print(words_keep_track[j], ": ", words_neighbors)




