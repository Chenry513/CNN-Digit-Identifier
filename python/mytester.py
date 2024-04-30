import numpy as np
from utils import get_lenet
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import cv2
import matplotlib.pyplot as plt

network_layer = get_lenet(5)
network_param = init_convnet(network_layer)
training_data = loadmat('../results/lenet.mat')
network_params_raw = training_data['params']
ibatch = np.zeros((28*28, 5))

for p in range(len(network_param)):
    rweights = network_params_raw[0,p][0,0][0]
    rbiases = network_params_raw[0,p][0,0][1]
    assert network_param[p]['w'].shape == rweights.shape, 'same shape error'
    assert network_param[p]['b'].shape == rbiases.shape, 'same shape error'
    network_param[p]['w'] = rweights
    network_param[p]['b'] = rbiases
for j in range(5):
    imag = cv2.imread('../myimages/%d.png' %j)
    imags = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    networkh = np.shape(imags)[0]
    networkw = np.shape(imags)[1]
    imags = np.reshape(imags, (-1, networkh*networkw))
    ibatch[:,j] = imags
for i in range(5):
    imag = ibatch[:,i]
    imag = np.reshape(imag, (28, 28))
    plt.imshow(imag, cmap='gray')
    plt.show()
cptest, P = convnet_forward(network_param, network_layer, ibatch, test = True)
preds = []
preds.append(np.argmax(P, axis = 0))
all_preds = np.reshape(preds, (1, np.shape(preds)[0] * np.shape(preds)[1]))
print(preds)






