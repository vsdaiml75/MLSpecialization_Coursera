import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

plt.style.use('./deeplearning.mplstyle')

# Training data
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x, y, w, b):
    """
    Computes the cost function for a linear regression model.
    
    Args:
        x (ndarray (m,)): Data, m examples 
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters  
    Returns:
        total_cost (float): The cost of the model
    """ 
    m = x.shape[0]

    # Initialize cost_sum to 0
    cost_sum = 0

    # Loop over all examples
    for i in range(m):
        # Compute the prediction of the ith example
        f_wb = w * x[i] + b

        # Add the squared error of the ith example to cost_sum
        cost_sum = cost_sum + (f_wb - y[i])**2

    # Compute the cost function
    total_cost = 1 / (2 * m) * cost_sum

    return total_cost

plt_intuition(x_train, y_train)

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 400, 500, 600, 650])

plt_intuition(x_train, y_train)

plt.close('all')

# Create a figure and axis
fig,ax,dyn_items = plt_stationary(x_train, y_train)

# Create an updater function that will be called when the user clicks on the plot
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

soup_bowl()




