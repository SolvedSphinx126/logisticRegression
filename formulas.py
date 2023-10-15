import numpy as np
import math

# This was harder to figure out than I thought
def sigmoid(z):
    if isinstance(z, list):
        return list(map(sigmoid, z))
    else:
        # Try catch is for overflow in exponentiation. If the exponent is too large, we can just return 0
        # becasue the number would have been 1 / (number so big it overflows) thus returning zero is the best approximation
        try:
            return 1.0 / (1.0 + (math.exp(-z)))
        except:
            return 0

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

#Creates the confusion matrix for each prediction.
def confusionMatrix(predictedData, actualData):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(predictedData)):
        if (predictedData[i] == 1):
            if (actualData[i] == 1):
                #True Positive
                tp+=1
            else:
                #False Positive
                fp+=1
        else:
            if (actualData[i] == 1):
                #False Negative
                fn+=1
            else:
                #True Negative
                tn+=1
    return tp, fp, tn, fn

#Accuracy Function
def accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + tn + fp + fn)

#Precision Function
def precision(tp, fp):
    return tp / (tp + fp)
