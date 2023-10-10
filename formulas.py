import numpy as np
import math

# This was harder to figure out than I thought
def sigmoid(z):
    if isinstance(z, np.ndarray):
        return np.vectorize(lambda x: 1 / (1 + (math.e ** -x)))(z)
    else:
        return 1 / (1 + (math.e ** -z))

def hypothesis(x, theta):
    if isinstance(theta, np.ndarray) & isinstance(x, np.ndarray):
        return sigmoid(np.matmul(theta.T, x))