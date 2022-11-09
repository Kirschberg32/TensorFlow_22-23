import numpy as np
import matplotlib.pyplot as plt
import utils

# Task 04 - Training
def training(model,x,t,epochs : int =  1000,visualize : bool = False,target_function = lambda x: x**3 - x**2):
    """ trains a given model over several epochs given the data

    Parameters: 
        model (MLP) = the model to train
        x (array) = the x value of the data
        t (array) = the target value of the data
        epochs (int) = the number of epochs to train for
        visualize(boolean) = if you want to visualize predictions of the model during training
        target_function = the function that created the target values (only needed for the visualization)
    """

    losses = list()

    if(visualize):
        visualize_prediction(x,model,"untrained",target_function)

    for e in range(epochs):
        loss_in_episode = list()

        for i, xi in enumerate(x): # for each data pair
            xi = np.array([[xi]])
            
            y  = model.forward_step(xi)

            error = utils.derivLoss(y[0,0],t[i])
            model.backpropagation(np.array([[error]]))

            loss_in_episode.append(utils.Loss(y[0,0],t[i]))

        losses.append(loss_in_episode)

    if(visualize):
        visualize_prediction(x,model," trained",target_function) # visualize the results of the last epoch

    average_loss = np.mean(losses,axis=1)
    return average_loss

# Task 05 - Visualization
def visualize_results(average_loss, epochs : int = 1000,):
    """ visualizes the average loss during training 
    
    Parameters: 
        average_loss (array/list) = average loss per epoch 
        epochs (int) = the number of epochs that have been run
    """

    plt.plot(range(epochs),average_loss)
    plt.title("Average loss")
    plt.xlabel('epochs') 
    plt.ylabel('loss') 
    plt.show()

def visualize_prediction(x,model,when : str, target_function = lambda x: x**3 - x**2):
    """ visualizes the model's predictions

    Parameters:
        x (array) = the x value
        model (MLP) = the model
        target_function = the function that created the target values
    """

    plot_x = np.linspace(0,1,100)
    plt.plot(plot_x, target_function(plot_x),c="red")
    y = list()
    for i, xi in enumerate(x): # for each data pair
        y.append( model.forward_step(np.array([[xi]])))

    plt.scatter(x,y)
    plt.title("Predictions of the mlp " + when)
    plt.xlabel('x') 
    plt.ylabel('predictions y') 
    plt.show()