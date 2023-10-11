# Import necessary libraries
import math
import pandas as pd
import numpy as np
import formulas
import scipy

irisDataFile = open("iris_data/iris.data")
datalines = irisDataFile.readlines()
rawDataVals = []

for line in datalines:
    for i in range(len(line.split(","))):
        rawDataVals.append((line.split(",")[i]).strip() if i == len(line.split(",")) - 1 else float(line.split(",")[i]))

dataLabels = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
formattedDataVals = []

i = 0
while i < len(rawDataVals):
    formattedDataVals.append([rawDataVals[j] for j in range(i, i + len(dataLabels))])
    i += len(dataLabels)

dataframe = pd.DataFrame(formattedDataVals, columns=dataLabels)

traningData = dataframe.sample(frac=.8)
testData = dataframe.drop(traningData.index)

# Try again if we got unlucky and didn't have an instance of all three classes in the validation set
if not ("Iris-setosa" in testData["class"].values and "Iris-versicolor" in testData["class"].values and "Iris-virginica" in testData["class"].values):
    print("Retrying selection of traning and validation data")
    traningData = dataframe.sample(frac=.8)
    testData = dataframe.drop(traningData.index)

def getTrainedThetas(dataframe, yLable):
    theta = np.zeros(len(dataLabels), np.float32)

    trainingX = dataframe[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    trainingX = np.array(trainingX.values)
    trainingX = np.concatenate(((np.ones((trainingX.shape[0], 1), dtype=trainingX.dtype)), trainingX), axis=1)

    trainingY = dataframe[["class"]]
    trainingY = trainingY.apply(lambda col: [0 if val == yLable else 1 for val in col], raw=True)
    trainingY = np.array(trainingY.values)

    res = scipy.optimize.minimize(formulas.cost, theta, (trainingX, trainingY), options={"disp": True})
    return res["x"]


trainedThetas = getTrainedThetas(traningData, "Iris-setosa")
resultData = []

validX = testData[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
validX = np.array(validX.values)
validX = np.concatenate(((np.ones((validX.shape[0], 1), dtype=validX.dtype)), validX), axis=1)
validY = testData[["class"]]
validY = validY.apply(lambda col: [0 if val == "Iris-setosa" else 1 for val in col], raw=True)
validY = np.array(validY.values)

for y in enumerate(validY):
    resultData.append(0 if formulas.hypothesis(validX[y[0]], trainedThetas) < 0.5 else 1)
    print(f"Result Data: {resultData[y[0]]}, Actual Data: {y[1]}, Equal: {resultData[y[0]] == y[1]}")