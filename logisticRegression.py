# Import necessary libraries
import math
import pandas as pd
import numpy as np
import formulas

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