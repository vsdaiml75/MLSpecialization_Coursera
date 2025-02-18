import numpy as np
import matplotlib.pyplot as plt

# Use a default style instead of the custom deep learning style
plt.style.use('./deeplearning.mplstyle')  # or try 'classic', 'seaborn', etc.

# Create training data for house prices (in 1000s of dollars)
# Input variable i.e. feature (size in 1000 square feet)
x_train = np.array([1.0, 2.0])
# Output variable i.e. target (price in 1000s of dollars)
y_train = np.array([300.0, 500.0])

w = 100
b = 100

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

print(f"w: {w}")
print(f"b: {b}")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')

# Set the title and labels
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')

# Display the plot
plt.show()

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns:
      f_wb (ndarray (m,)): target values (predicted)
    """
    # Get the number of examples in input array x
    m = x.shape[0]
    # Initialize the prediction array f_wb with zeros
    f_wb = np.zeros(m)
    # Loop through each example in x
    for i in range(m):
        # Compute the prediction for the i-th example
            f_wb[i] = w * x[i] + b
    # Return the prediction array f_wb
    return f_wb

# Compute the prediction for the training data
temp_f_wb = compute_model_output(x_train, w, b)

# Plot the prediction
plt.plot(x_train, temp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')

# Set the title and labels
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')

# Display the legend
plt.legend()   
# Display the plot
plt.show()

w=200
b=100
x_i = 1.2

cost_1200sqft = w * x_i + b

print(f"The cost for a house with 1200 sqft is ${cost_1200sqft*1000:0.0f}") 



