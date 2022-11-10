from data_creator import create_dataset
from MLP import MLP
from training_visualization import training, visualize_results

import numpy as np

if __name__ == "__main__":

    tf_list = [lambda x: x**3 - x**2,lambda x:x**2,lambda x: np.sin(1/x)]
    tf = tf_list[0] # choose the target function here

    x,t = create_dataset(size = 100, target_function = tf, plot = True,equal_spaced = False)

    # create the model with one hidden layer (length of list), having 10 units (value of element)
    mlp = MLP(hidden_layers = [10,],input_units = 1,output_units = 1)

    epochs_number = 1000
    average_loss = training(mlp,x,t,epochs = epochs_number,visualize = True,target_function = tf) # target function only needed for visualization (predictions of the model)
    
    visualize_results(average_loss,epochs = epochs_number)