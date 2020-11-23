#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import math
import random
import sys
import os
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext


def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setAppName("Logistic Regression With Simple GD")) # Name of App
    sc = SparkContext(conf = conf) 
    return sc

sc = getSparkContext()


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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gd_func(point, w):
    x, y = point.features, point.label
    x = np.array(x)
    error = sigmoid(w.T.dot(x))
    return (error - y) * x

def gd(dataset, w):
    g = dataset.map(lambda point: gd_func(point, w)).sum()
    return g

def cost_func(point, w):
    x, y = point.features, point.label
    x = np.array(x)
    error = sigmoid(w.T.dot(x))
    return abs(y - error)

def cost(dataset, w):
    total_cost = dataset.map(lambda point: cost_func(point, w)).sum()
    return total_cost

def logistic_gd(dataset):
    w = np.zeros(4)
    limit = 200
    eta = 0.1
    costs = []
    for i in range(limit):
        current_cost = cost(dataset, w)
        costs.append(current_cost)
        w = w - eta * gd(dataset, w)
        eta = eta * 0.95
    return w

def predict(x, w):
    x = np.array(x)
    l = sigmoid(w.T.dot(x))
    predicted_val = round(l)

    return int(predicted_val)

w = logistic_gd(parsedData)
labelsAndPreds = parsedData.map(lambda point: (point.label, predict(point.features, w)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())

print("Training Error = " + str(trainErr))
