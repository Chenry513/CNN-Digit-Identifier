import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

##### Fill in your code here to plot the features ######
network_layer = get_lenet()
network_param = init_convnet(network_layer)
training_data = loadmat('../results/lenet.mat')
network_params_raw = training_data['params']

for params in range(len(network_param)):
    rweights = network_params_raw[0,params][0,0][0]
    rbiases = network_params_raw[0,params][0,0][1]
    assert network_param[params]['w'].shape == rweights.shape, 'same shape error'
    assert network_param[params]['b'].shape == rbiases.shape, 'same shape error'
    network_param[params]['w'] = rweights
    network_param[params]['b'] = rbiases

set = False
xtrain_val, ytrain_val, xvalidate_val, yvalidate_val, xtest_val, ytest_val = load_mnist(set)
m_train_val = xtrain_val.shape[1]
batch_size = 1
network_layer[0]['batch_size'] = batch_size
imag = xtest_val[:,0]
imag = np.reshape(imag, (28, 28), order='F')
plt.imshow(imag.T, cmap='gray')
plt.show()

output = convnet_forward(network_param, network_layer, xtest_val[:,0:1])
outputS = np.reshape(output[0]['data'], (28,28), order='F')
conv_out = np.reshape(output[1]['data'].T, (24, 24, -1), order='F')  
relu_out = np.reshape(output[2]['data'].T, (24, 24, -1), order='F')  

p1, axis1 = plt.subplots(4, 5, figsize=(12, 10))
p1.suptitle("Convolution")
for p in range(min(4 * 5, conv_out.shape[2])):
    row = p // 5
    col = p % 5
    axis1[row, col].imshow(conv_out[:, :, p], cmap='gray', aspect='auto')
    axis1[row, col].axis('off')
plt.tight_layout()
p1.savefig("../results/convolution.png", dpi=400)

p2, axis2 = plt.subplots(4, 5, figsize=(12, 10))
p2.suptitle("ReLU")
for j in range(min(4 * 5, relu_out.shape[2])):
    row = j // 5
    col = j % 5
    axis2[row, col].imshow(relu_out[:, :, j], cmap='gray', aspect='auto')
    axis2[row, col].axis('off')
plt.tight_layout()
p2.savefig("../results/relu.png", dpi =400)
#plt.show()

