import data_processing as dp

# variables from prepare, do not change, or otherwise prepare newly
original_file_path = "data/bible.txt"
prepared_file_path = "data/prepared_bible.txt"
model_prefix = 'tokenizer_model'
VOCABUALRY_SIZE = 2000
WINDOW_SIZE = 64
BATCH_SIZE = 64

# prepare if you want to create a new tokenizer and a new prepared data file
data, prepared_data, tokenizer = dp.prepare_everything(original_file_path,prepared_file_path,model_prefix,VOCABUALRY_SIZE)

# if you only want to create a new tokenizer first, use loading afterwards
# dp.train_tokenizer(prepared_file_path,VOCABUALRY_SIZE,model_prefix)

# load everything if already prepared
# data, prepared_data, tokenizer = dp.prepare_everything(original_file_path,prepared_file_path,model_prefix)

dataset = dp.create_dataset(prepared_data,tokenizer,WINDOW_SIZE, BATCH_SIZE)

