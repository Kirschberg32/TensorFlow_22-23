# Task 01 - Building your data set
import numpy as np
import matplotlib.pyplot as plt

# Task 01 - Building your data set
def create_dataset(size : int = 100,target_function = lambda x: x**3 - x**2,plot:bool = False,equal_spaced : bool = False):
    """ creates a dataset given a target_function

    Parameters: 
        size = the size of the dataset
        target function = the function to create the target values
        plot (boolean) = if you want to plot the data after creating it
        equal_spaced (boolean) = Set to True if you want equally spaced data point, otherwise you get uniformly distributed ones
    """
    start = 0.0000001 # to prevent 0 division
    if (equal_spaced):
        x = np.linspace(start ,1,size) # equally spaced values
    else:
        x = np.random.uniform(start,1,(size)) # random values

    t = target_function(x)

    if(plot):
        plot_dataset(x,t,target_function)

    return x,t


# Optional: Plot your data points 
# along with the underlying function which generated them
def plot_dataset(x,t,target_function = lambda x: x**3 - x**2):
    """ plots a by create_dataset created dataset and the target function which created it

    Parameters:
        x = the x value of the dataset
        t = the target value of the dataset
        target_function = the target function which created the dataset
     """

    plot_x = np.linspace(0,1,100)
    plt.plot(plot_x, target_function(plot_x),c="red")
    plt.scatter(x,t)
    plt.title("Data")
    plt.xlabel('x') 
    plt.ylabel('t') 
    plt.show()