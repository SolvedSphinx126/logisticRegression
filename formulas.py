import numpy as np
import math

# This was harder to figure out than I thought
def sigmoid(z):
    if isinstance(z, np.ndarray):
        return np.vectorize(lambda x: 1 / (1 + (math.e ** -x)))(z)
    else:
        return 1 / (1 + (math.e ** -z))

def hypothesis(x, theta):
    if isinstance(theta, np.ndarray) and isinstance(x, np.ndarray):
        return min(max(sigmoid(np.matmul(theta.T, x)), 0.000000000001), 0.99999999999)

def cost(theta, data, y):
    curr_cum = 0
    m = len(data)
    for x in enumerate(data):
        #print(hypothesis(x[1], theta))
        curr_cum += -y[x[0]][0] * math.log(hypothesis(x[1], theta), 10) - (1 - y[x[0]][0]) * math.log(1 - hypothesis(x[1], theta), 10)
    return (1 / m) * curr_cum

