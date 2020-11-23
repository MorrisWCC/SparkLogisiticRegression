from __future__ import print_function
import sys
import os
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext

def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setAppName("Logistic Regression Sample Code")) # Name of App
    sc = SparkContext(conf = conf) 
    return sc

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("gs://dataproc-staging-us-central1-567425075225-qwbh88hv/data_banknote_authentication.txt")

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD, it's in the end in our data, so 
    # putting it in the right place
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    # feats.insert(0,label)
    features = [ float(feature) for feature in feats ] # need floats
    return LabeledPoint(label, features)

parsedData = data.map(mapper)
model = LogisticRegressionWithSGD.train(parsedData, iterations=200, step=0.1)

# Predict the first elem will be actual data and the second 
# item will be the prediction of the modelS"
labelsAndPreds = parsedData.map(lambda point: (point.label, model.predict(point.features)))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))

