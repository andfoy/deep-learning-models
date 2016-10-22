from __future__ import division

import os
import cost
import minimize
import numpy as np



def flatten_matrix(mat):
    mat = mat.flatten(1)
    mat = mat.reshape(len(mat), 1, order='F')
    return mat


def initializeParameters(outputSize, hiddenSize, visibleSize):
    r = np.sqrt(6) / np.sqrt(hiddenSize+visibleSize+1)
    W1 = np.random.rand(hiddenSize, visibleSize) * 2 * r - r
    W2 = np.random.rand(outputSize, hiddenSize) * 2 * r - r
    b1 = np.zeros((hiddenSize, 1))
    b2 = np.zeros((outputSize, 1))
    return np.vstack([flatten_matrix(W1), flatten_matrix(W2), b1, b2])

def preProcess(inputs, epsilon):
    m = inputs.shape[1]
    meanInput = np.mean(inputs, 1)
    meanInput = flatten_matrix(meanInput)
    inputs = inputs-meanInput
    sigma = np.dot(inputs, inputs.T)/m
    U,s,V = np.linalg.svd(sigma)
    S = np.zeros(V.shape)
    for i in range(0, V.shape[0]):
        S[i,i] = s[i]
    ZCAWhite = np.dot(np.dot(U, np.diag(1 / np.sqrt(np.diag(S) + epsilon))), U.T) 
    inputs = np.dot(ZCAWhite, inputs)
    return [inputs, meanInput, ZCAWhite]

def process_data(inputs, values):
    _beta = 2
    _lambda = 1e-4
    _epsilon = 0.1
    _sparsityParam = 0.6
    num_iter = 7000

    inputSize = inputs.shape[0]
    m = inputs.shape[1]
    hiddenSize = 180
    outputSize = 1

    theta = initializeParameters(outputSize, hiddenSize, inputSize)
    inputs, meanInput, ZCAWhite = preProcess(inputs, _epsilon)
    costF = lambda p: cost.sparseLinearNNCost(p, inputSize, hiddenSize, outputSize, _lambda, _sparsityParam, _beta, inputs, values)

    optTheta,costV,i = minimize.minimize(costF,theta,maxnumlinesearch=num_iter)
    pred = cost.predict(inputs, optTheta, inputSize, hiddenSize, outputSize)

    diff = np.linalg.norm(pred-values)/np.linalg.norm(pred+values)

    print "Total RMSE: %g" % (diff)
    print "Saving parameters..."

    np.savez('parameters.npz', optTheta = optTheta, meanInput = meanInput, ZCAWhite = ZCAWhite)

    print "Done!"

def predict(inputs):
    visibleSize = inputs.shape[0]
    hiddenSize = 180
    outputSize = 1
    parameters = np.load('parameters.npz')
    meanInput = parameters['meanInput']
    ZCAWhite = parameters['ZCAWhite']
    optTheta = parameters['optTheta']
    inputs = inputs - meanInput
    inputs = np.dot(ZCAWhite, inputs)
    values = cost.predict(inputs, optTheta, visibleSize, hiddenSize, outputSize)
    return values


