import numpy as np
from utils import get_lenet
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import cv2
import matplotlib.pyplot as plt

image1correct = [5, 9, 6, 8, 7, 0, 3, 4, 2, 1]
image2correct = [9, 6, 7, 3, 8, 0, 5, 4, 2, 1]
image3correct= [2, 0, 6, 6, 6]
image4correct = [2, 1, 0, 4, 1, 4, 5, 7, 9, 9, 6, 0, 0, 1, 5, 3, 9, 4, 9, 7, 6, 6, 0, 1, 5, 4, 4, 0, 9, 7, 1, 3, 4, 2, 1, 2, 1, 3, 7, 7, 2, 1, 2, 3, 1, 5, 4, 7, 4, 4]

total = 0
totalI = 0
for i in range(4):
    thresh= 127
    if (i+1) != 3:
        img = cv2.imread('../images/image%d.jpg' % (i+1))
    else:
        img = cv2.imread('../images/image%d.png' % (i+1))
    if i+1 == 3:
        thresh = 0
    imggr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, img4cnn = cv2.threshold(imggr, 127, 255, cv2.THRESH_BINARY_INV)
    ret, imgth = cv2.threshold(imggr, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    
    if i+1 == 4:
        imgth = cv2.dilate(imgth, kernel, iterations=1)
   
    if i+1 == 3:
        imgth = cv2.erode(imgth, kernel, iterations=1)
    output = cv2.connectedComponentsWithStats(imgth, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    newImage = img.copy()

    all_preds = []

    for x in range(1, numLabels):  
        left = stats[x, cv2.CC_STAT_LEFT]
        top = stats[x, cv2.CC_STAT_TOP]
        w = stats[x, cv2.CC_STAT_WIDTH]
        h = stats[x, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(newImage, (left, top), (left + w, top + h), (0, 0, 255), 5)

    
    plt.imshow(cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))
    plt.title(f'Image {i+1} Bounding Boxes')

    if i+1 == 1:
        imageValid = image1correct
    elif i + 1 == 2:
        imageValid = image2correct
    elif i + 1 == 3:
        imageValid = image3correct
    elif i + 1 == 4:
        imageValid = image4correct

    input_data = np.zeros((28*28, numLabels-1))
    totalI += numLabels - 1
    network_layer = get_lenet(numLabels-1)
    network_param = init_convnet(network_layer)
    training_data = loadmat('../results/lenet.mat')
    network_params_raw = training_data['params']
    for p in range(len(network_param)):
        rweights = network_params_raw[0, p][0, 0][0]
        rbiases = network_params_raw[0, p][0, 0][1]
        assert network_param[p]['w'].shape == rweights.shape, 'same shape error'
        assert network_param[p]['b'].shape == rbiases.shape, 'same shape error'
        network_param[p]['w'] = rweights
        network_param[p]['b'] = rbiases

    for a in range(1, numLabels):
        left = stats[a, cv2.CC_STAT_LEFT]
        top = stats[a, cv2.CC_STAT_TOP]
        w = stats[a, cv2.CC_STAT_WIDTH]
        h = stats[a, cv2.CC_STAT_HEIGHT]
        curr_img = img4cnn[top:top+h, left:left+w]
        imgh = np.shape(curr_img)[0]
        imgw = np.shape(curr_img)[1]
        
        if imgh >= 2*imgw:
            curr_img = np.pad(curr_img, (round(imgh * 0.25), round(imgw * 0.35)), 'constant', constant_values=0)
       
        elif imgw >= 2*imgh:
            curr_img = np.pad(curr_img, (round(imgh * 0.35), round(imgw * 0.25)), 'constant', constant_values=0)
        
        else:
            curr_img = np.pad(curr_img, (round(imgh * 0.3), round(imgw * 0.3)), 'constant', constant_values=0)
        curr_img = cv2.resize(curr_img, (28, 28))
        input_data[:, a-1] = np.reshape(curr_img, (1, -1))

    cpTest, P = convnet_forward(network_param, network_layer, input_data, test=True)

    
    all_preds = np.argmax(P, axis=0).tolist()

    for a in range(1, numLabels):
        left = stats[a, cv2.CC_STAT_LEFT]
        top = stats[a, cv2.CC_STAT_TOP]
        w = stats[a, cv2.CC_STAT_WIDTH]
        h = stats[a, cv2.CC_STAT_HEIGHT]
        predicted_label = all_preds[a - 1]  
        label_x = left + w // 2 - 10  
        label_y = top + h // 2 + 10
        plt.text(label_x, label_y, str(predicted_label), fontsize=12, color='green')

    plt.show() 

    













