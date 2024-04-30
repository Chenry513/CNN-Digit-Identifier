import numpy as np
import os
import matplotlib.pyplot as plt
from conv_layer import conv_layer_forward
from pooling_layer import pooling_layer_forward
from inner_product import inner_product_forward

resultsdir = '../results'
os.makedirs(resultsdir, exist_ok=True)

def test_pooling_1():
    input_data = {'data': np.zeros((36*3,2))}
    input_data['data'][12, 0] = 0.5
    input_data['data'][13, 0] = 0.25
    input_data['data'][14, 0] = 0.5
    input_data['data'][19+72, 0] = 0.75

    input_data['data'][14, 1] = 0.25
    input_data['data'][15, 1] = 0.75
    input_data['data'][5+36, 1] = 0.75
    input_data['data'][11+72, 1] = 0.75
    input_data['width'] = 6
    input_data['height'] = 6
    input_data['channel'] = 3
    input_data['batch_size'] = 2

    layer = {'type': 'POOLING', 'k': 2, 'stride': 2, 'pad': 0}

    output = pooling_layer_forward(input_data, layer)
    display_results(input_data, output, 'Pooling Test')


def test_inner_1():
    # Initialize the 'input' structure
    input_data = {}
    input_data['data'] = np.zeros((25,2))
    input_data['data'] = input_data['data'].T
    input_data['data'].flat[5::3] = 1.0
    input_data['data'].flat[6::3] = 0.5
    input_data['data'] = input_data['data'].T
    input_data['height'] = 25
    input_data['width'] = 1
    input_data['channel'] = 1
    input_data['batch_size'] = 2

    # Initialize the 'layer' structure
    layer = {}
    layer['type'] = 'IP'
    layer['num'] = 25

    # Initialize the 'params' structure
    params = {}
    params['w'] = np.eye(25)
    params['w'].flat[:25*10] = 0
    params['w'][1, 4] = 0.5
    params['w'][2, 3] = 0.5
    params['b'] = np.zeros((1,25))
    params['b'][0,1] = 0.5
    params['b'][0,3] = 0.5

    output = inner_product_forward(input_data, layer, params)

    display_results_2(input_data, output, params, 'Inner Product Test')


def test_conv_1():
    # Initialize the 'input' structure
    input_data = {}
    input_data['data'] = np.zeros((25, 2))
    input_data['data'][12, 0] = 1
    input_data['data'][13, 1] = 1
    input_data['width'] = 5
    input_data['height'] = 5
    input_data['channel'] = 1
    input_data['batch_size'] = 2

    # Initialize the 'conv_layer' structure
    conv_layer = {}
    conv_layer['type'] = 'CONV'
    conv_layer['num'] = 3
    conv_layer['k'] = 5
    conv_layer['stride'] = 1
    conv_layer['pad'] = 2

    # Initialize the 'params' structure
    params = {}
    params['w'] = np.zeros((25, 3))
    params['w'][13, 0] = 0.5  # move image left by one pixel on red channel
    params['w'][11+5, 2] = 0.5  # move image top-right dir on blue channel
    params['b'] = np.array([0.25, 0.0, 0.25])

    # Call the conv_layer_forward function (you would need to define this function in Python)
    output = conv_layer_forward(input_data, conv_layer, params)

    # Call the display_results function (you would need to define this function in Python)
    display_results(input_data, output, 'Convolution Test 1')


def test_conv_2():
    # Initialize the 'input' structure
    input_data = {}
    input_data['data'] = np.zeros((75, 4))

    input_data['data'][12, 0] = 1
    input_data['data'][12+25, 1] = 1
    input_data['data'][13+50, 2] = 1

    input_data['data'][0, 3] = 1
    input_data['data'][21, 3] = 1
    input_data['data'][12, 3] = 1
    input_data['data'][13, 3] = 1
    input_data['data'][13+25, 3] = 1
    input_data['data'][14+25, 3] = 1
    input_data['data'][24+50, 3] = 1

    input_data['width'] = 5
    input_data['height'] = 5
    input_data['channel'] = 3
    input_data['batch_size'] = 4

    # Initialize the 'conv_layer' structure
    conv_layer = {}
    conv_layer['type'] = 'CONV'
    conv_layer['num'] = 3
    conv_layer['k'] = 5
    conv_layer['stride'] = 1
    conv_layer['pad'] = 2

    # Initialize the 'params' structure
    params = {}
    params['w'] = np.zeros((75, 3))
    # What it does to red
    params['w'][13, 0] = 1.  # move image left by one pixel on red channel
    params['w'][11+5, 2] = 1.  # move image top-right dir on blue channel
    # What it does to green
    params['w'][12+25, 2] = 1.  # stay in place on blue
    params['w'][12+5+25, 1] = 1.    # move top on green
    # What it does to blue
    params['w'][12+50, 0] = 1.0  # stay in place
    params['w'][12+50, 1] = 1.0  # stay in place
    params['w'][12+50, 2] = 1.0  # stay in place
    # Bias
    params['b'] = np.array([0., 0.0, 0.])

    # Call the conv_layer_forward function (you would need to define this function in Python)
    output = conv_layer_forward(input_data, conv_layer, params)

    # Call the display_results function (you would need to define this function in Python)
    display_results(input_data, output, 'Convolution Test 2')


def display_results(input_data, output, testname):
    global resultsdir
    
    fig, ax = plt.subplots(input_data['batch_size'], 2)
    for batch in range(input_data['batch_size']):
        # outputs
        img1 = output['data'][:,batch].reshape(output['channel'], output['height'], output['width'])
        img1 = np.transpose(img1, (1, 2, 0))
        ax[batch, 1].imshow(img1)
        ax[batch, 1].set_title(f'Output {batch + 1}')
        ax[batch, 1].set_axis_off()

        # inputs
        imgin1 = input_data['data'][:,batch].reshape(input_data['channel'], input_data['height'], input_data['width'])
        imgin1 = np.transpose(imgin1, (1, 2, 0))
        ax[batch, 0].imshow(imgin1)
        ax[batch, 0].set_title(f'Input {batch + 1}')
        ax[batch, 0].set_axis_off()

    fig.suptitle(testname)
    
    filename = f"{resultsdir}/{testname}.png"
    plt.savefig(filename)


def display_results_2(input_data, output, params, testname):
    global resultsdir
    
    fig, ax = plt.subplots(input_data['batch_size'], 1)
    for batch in range(input_data['batch_size']):
        # outputs
        img = output['data'][:,batch].reshape(output['height'], output['width'])
        
        # middle
        img = np.hstack([params['w'].T, np.ones(img.shape), params['b'].reshape(-1, 1), np.zeros(img.shape), img])
        
        # inputs
        imgin = input_data['data'][:,batch].reshape(input_data['height'], input_data['width']).T
        imgin_padded = np.hstack([imgin, np.zeros((imgin.shape[0], 4))])
        img = np.vstack([imgin_padded, np.zeros(imgin_padded.shape), np.zeros(imgin_padded.shape), np.ones(imgin_padded.shape), img])
        
        ax[batch].imshow(img)
        ax[batch].set_title(f'Batch {batch + 1}')
        ax[batch].set_axis_off()
        
    fig.suptitle(testname)
    
    filename = f"{resultsdir}/{testname}.png"
    plt.savefig(filename)


test_conv_1()
test_conv_2()
test_pooling_1()
test_inner_1()
