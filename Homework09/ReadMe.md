# Files: 
* [logs](logs): contains the tensorboard logs for all the runs
* [Plots](Plots): contains all saved plots
* [main.py](main.py): only used for execution and creating all the analysis and plots. 
* [model.py](model.py): creation of MyCNNBlock, MyCNNNormalizationLayer, MyCNN as an Discriminator and MyGenerator
* [get_data.py](get_data.py): contains all functions to load and preprocess the qick draw dataset
* [training_loop.py]: contains the training
* [saved_discriminator](saved_encoder): contains all the saved discriminators
* [saved_generator](saved_decoder): contains the saved generators

<img src="Plots/run-1.png" align="left" alt="Plot of training results with 10 epochs in run 1" width="700"/>
Plot of training results with 10 epochs in run 1. The Embedding was of size 10. 
<br clear="left"/>

# Latent Space Analysis
<img src="Plots/run-1_embedding.png" align="left" alt="Plot of the embeddings" width="700"/>
The embeddings for the first 1000 validation set images, reduced to two dimensions. 
<br clear="left"/>
It does look close to what we expected, as we know that normal autoencoders do not tend to put different classes close together, but can have big gaps inbetween them. The labels 0, 1, 2, 6 and 7 are well seperated. The rest is still more mixed than what we would have liked after training. But with numbers as 3 and 8 is was expected, that they are difficult to distinguish.
<img src="Plots/run-1_interpolation.png" align="left" alt="Plot of an interpolation between two embeddings" width="700"/>
An Interpolation between two embeddings created by the decoder. 
<br clear="left"/>
