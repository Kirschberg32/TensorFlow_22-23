# Files: 

* [data](data): contains text files to work on
* [logs](logs): contains the tensorboard logs for all the runs
* [model](model): contains the trained model
* [plots](plots): contains all saved plots
* [text](text): contains output of each epoch while training using a prompt
* [main.py](main.py): only used for execution and creating all the analysis and plots.
* [main.ipynb](main.ipynb): Same as main.py. Just for visualization without executing it
* [model.py](model.py): creation of the Model. Contains a Embedder and a TransformerBlock
* [prepare.py](prepare.py): contains all functions to load and preprocess the text data
* [tokenizer_model]: contains model created by using the SentencePiece tokenizer

The Output is far from optimal. Probably because of our parameters and the amount of data used 
which already took about 5min per epoch and therefore was time-consuming.
For comparison, using the bible.txt from homework10 needed 45min per epoch using the same parameters.

Letting Tensorflow use the GPU could speed the progress significantly.
