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

theta = np.zeros(len(dataLabels), np.float32)
xs = traningData[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
xs = np.array(xs.values)
xs = np.concatenate(((np.ones((xs.shape[0], 1), dtype=xs.dtype)), xs), axis=1)
ys = traningData[["class"]]
ys = ys.apply(lambda col: [0 if val == "Iris-setosa" else 1 for val in col], raw=True)
ys = np.array(ys.values)
#print(xs[0])
#print(theta)
res = scipy.optimize.minimize(formulas.cost, theta, (xs, ys), options={"disp": True})
print(res["x"])