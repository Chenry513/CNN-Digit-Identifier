import numpy as np
from load_mnist import load_mnist
from init_convnet import init_convnet
from conv_net import conv_net
from utils import sgd_momentum, get_lr, get_lenet
import copy
from scipy.io import savemat

# Set random seeds
np.random.seed(100000)

# Network definition
layers = get_lenet()

# Loading data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)
xtrain = np.hstack((xtrain, xvalidate))
ytrain = np.hstack((ytrain, yvalidate))
m_train = xtrain.shape[1]
batch_size = 100

# Parameters initialization
mu = 0.9
epsilon = 0.01
gamma = 0.0001
power = 0.75
weight_decay = 0.0005
w_lr = 1
b_lr = 2

test_interval = 500
display_interval = 50
snapshot = 500
max_iter = 2001

# Use the following to train from scratch
params = init_convnet(layers)

params_winc = copy.deepcopy(params)

# Training the network
new_order = np.random.permutation(m_train)
xtrain = xtrain[:, new_order]
ytrain = ytrain[:, new_order]
curr_batch = 0  

for iter in range(max_iter):
    if curr_batch >= m_train:
        new_order = np.random.permutation(m_train)
        xtrain = xtrain[:, new_order]
        ytrain = ytrain[:, new_order]
        curr_batch = 0

    x_batch = xtrain[:, curr_batch:curr_batch+batch_size]
    y_batch = ytrain[:, curr_batch:curr_batch+batch_size]
    curr_batch += batch_size
    
    cp, param_grad = conv_net(params, layers, x_batch, y_batch)

    for l_idx in range(len(layers)-1):
        w_rate = get_lr(iter, epsilon * w_lr, gamma, power)
        w_params, w_params_winc = sgd_momentum(w_rate, mu, weight_decay, params, params_winc, param_grad)

        b_rate = get_lr(iter, epsilon * b_lr, gamma, power)
        b_params, b_params_winc = sgd_momentum(b_rate, mu, weight_decay, params, params_winc, param_grad)

        params[l_idx]['w'] = w_params[l_idx]['w']
        params_winc[l_idx]['w'] = w_params_winc[l_idx]['w']
        params[l_idx]['b'] = b_params[l_idx]['b']
        params_winc[l_idx]['b'] = b_params_winc[l_idx]['b']

    if iter % display_interval == 0:
        print(f"cost = {cp['cost']} training_percent = {cp['percent']}")

    if iter % test_interval == 0:
        print(iter)
        layers[0]['batch_size'] = xtest.shape[1]
        cptest, _ = conv_net(params, layers, xtest, ytest, test=True)
        layers[0]['batch_size'] = batch_size
        print(f"test accuracy: {cptest['percent']}")

    if iter % snapshot == 0:
        savemat('../results/lenet.mat', {'params': params})

