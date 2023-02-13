import data_processing as dp



if __name__ is "main":
    # variables, make sure they are the same as later when you load the 
    # prepared data adn tokenizer in the main
    original_file_path = "data/bible.txt"
    prepared_file_path = "data/prepared_bible.txt"
    model_prefix = 'tokenizer_model'
    VOCABUALRY_SIZE = 2000

    # if it is the first time and you do not have the data yet
    data = dp.load_data(original_file_path)
    prepared_data = dp.prepared_data(data,prepared_file_path)
    dp.train_tokenizer(prepared_file_path,VOCABUALRY_SIZE,model_prefix)