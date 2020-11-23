#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time

start = time.time()
#網路上找的dataset 可以線性分割
f = open('data.txt', 'r')
data = []
lines = f.readlines()

for line in lines:
    line = line.split(',')
    feat = [float(val) for val in line[:-1]]
    data.append(((feat), int(line[-1].split('\n')[0])))

dataset = np.array(data)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sgd(dataset, w):
    #run sgd randomly
    ind = random.randint(0, len(dataset) - 1)
    x, y = dataset[ind]
    x = np.array(x)
    error = sigmoid(w.T.dot(x))
    g = (error - y) * x
    return g

def gd(dataset, w):
    g = 0
    for x, y in dataset:
        x = np.array(x)
        error = sigmoid(w.T.dot(x))
        g += (error - y) * x
    
    return g

def cost(dataset, w):
    total_cost = 0
    for x,y in dataset:
        x = np.array(x)
        error = sigmoid(w.T.dot(x))
        total_cost += abs(y - error)
    return total_cost

def logistic_gd(dataset):
    w = np.zeros(4)
    limit = 200
    eta = 0.01
    costs = []
    for i in range(limit):
        current_cost = cost(dataset, w)
        costs.append(current_cost)
        w = w - eta * gd(dataset, w)
        eta = eta * 0.95
    return w

w2 = logistic_gd(dataset)

rate = 0
for feat, label in dataset:
    l = 0
    for i in range(len(w2)):
        l += w2[i] * feat[i]
    a = round(sigmoid(l))
    label = int(label)

    if a == label:
        rate += 1

print((1 - (rate / len(dataset))))
print(f"Elapsed Time: {time.time() - start}")