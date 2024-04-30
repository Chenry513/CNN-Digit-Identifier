import numpy as np
from scipy.io import loadmat

def load_mnist(fullset=True):
    data = loadmat('../images/mnist_all.mat')
    
    xtrain = np.vstack([data['train' + str(i)] for i in range(10)])
    ytrain = np.concatenate([i * np.ones((data['train' + str(i)].shape[0], 1)) for i in range(10)])
    
    xtest = np.vstack([data['test' + str(i)] for i in range(10)])
    ytest = np.concatenate([i * np.ones((data['test' + str(i)].shape[0], 1)) for i in range(10)])
    
    xtrain = xtrain.astype(np.float64) / 255
    xtest = xtest.astype(np.float64) / 255
    
    p_train = np.random.permutation(xtrain.shape[0])
    p_test = np.random.permutation(xtest.shape[0])
    
    xtrain, ytrain = xtrain[p_train], ytrain[p_train]
    xtest, ytest = xtest[p_test], ytest[p_test]
    
    m_validate = 10000
    xvalidate, yvalidate = xtrain[:m_validate], ytrain[:m_validate]
    xtrain, ytrain = xtrain[m_validate:], ytrain[m_validate:]
    
    xtrain, ytrain = xtrain.T, ytrain.T
    xvalidate, yvalidate = xvalidate.T, yvalidate.T
    xtest, ytest = xtest.T, ytest.T
    
    if not fullset:
        factor = 20
        xtrain, ytrain = xtrain[:, :xtrain.shape[1]//factor], ytrain[:, :ytrain.shape[1]//factor]
        xvalidate, yvalidate = xvalidate[:, :xvalidate.shape[1]//factor], yvalidate[:, :yvalidate.shape[1]//factor]
        xtest, ytest = xtest[:, :xtest.shape[1]//factor], ytest[:, :ytest.shape[1]//factor]
    
    return xtrain, ytrain, xvalidate, yvalidate, xtest, ytest

