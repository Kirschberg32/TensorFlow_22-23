# Files: 
* [logs](logs): contains the logs for all the test runs
* [csvs](csvs): contains csvs with the results from the test runs
* [Plots](Plots): contains images of the plots
* [main.py](main.py): only used for execution
* [training_loop.py](training_loop.py): contains one full training loop and the LogWriter
* [model.py](model.py): creation of MyCNNNormalizationLayer, MyCNNBlock and MyCNN class, containing train step and test step and metrices
* [get_data.py](get_data.py): contains all functions to load, preprocess and augment the data

# Overfitting: 
We already tried to minimize overfitting in last weeks homework by choosing a simple architecture with good expressivity (best amount of parameters). Because this also optimizes leanability and performance. 
We also normlized the data back then. <br />

<img src="Plots/Original/20221210-165137.png" align="left" alt="Plot of training results with original architecture" width="700"/>
You can see a slight overfitting, as there is a big difference in the curves of training and validation after about step 6. 
<br clear="left"/>

1. Data Augmentation: to create more data to learn from so less overfitting to fewer examples happens.
<img src="Plots/Aug/20221210-192125.png" align="middle" alt="Plot of training results with data augmentation" width="700"/>

2. DenseNet / ResNet: Skip connections help to improve the vanishing gradients problem and are specifically helpful for larger networks. The earlier input is concatenated (DenseNet) or added (ResNet) to the output. We did that for every block, not for every layer using ...
<img src="Plots/DenseMode/20221210-210143.png" align="left" alt="Plot of training results using skip connections DenseNet" width="700"/>
DenseNet architecture
<br clear="left"/>
<img src="Plots/ResMode/20221210-214111.png" align="left" alt="Plot of training results using skip connections ResNet" width="700"/>
ResNet architecture.
<br clear="left"/>

3. BatchNormalization layers: Avoids overfitting, provides regularization and can improve learning speed. It normalizes the output of one layer before it is given to the next layer. 
<img src="Plots/Normalisation/20221210-195043.png" align="left" alt="Plot of training results using batch normalization" width="700"/>
Batch Normalization does not work very well and seem to increase the amount of overfitting. 
<br clear="left"/>

4. Dropout layers: Prevents overfitting. During training it randomly switches some percentage of neurons of network on and off. 
<img src="Plots/Dropout_0.5/20221210-234117.png" align="left" alt="Plot of training results using droupout layers with a dropout rate of 0.5" width="700"/>
Using droupout layers with a dropout rate of 0.5. Seems to take more time to learn, which is expected as not all information is forwareded when training. 
<br clear="left"/>

5. L2 Regularizer: Regularizes the loss by controlling the models complexity. We decided to use L2, because when using L1 you get small parameter values, because the absolute value is reduced. With L2 the square is minimized and therefore the biggest values are reduced. 
<img src="Plots/Reg_L2/20221210-173112.png" align="left" alt="Plot of training results using an L2 regularizer" width="700"/>
Using an L2 regularizer. 
<br clear="left"/>

The following run contains all optimizations except for batch normalization as it seems to make the overfitting worse. As Skip connections it uses a ResNet architecture.

<img src="Plots/all_noBatchNorm/20221211-164311.png" align="middle" alt="Plot of training results with all architectures" width="700"/>