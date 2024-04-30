import numpy as np
from conv_layer import conv_layer_forward, conv_layer_backward
from pooling_layer import pooling_layer_forward, pooling_layer_backward
from inner_product import inner_product_forward, inner_product_backward
from relu import relu_forward, relu_backward
from mlrloss import mlrloss


def conv_net(params, layers, data, labels, test=False):
    """
    Convolutional neural network forward and backward pass.

    Parameters:
    - params (list of dicts): Contains the weights and biases for the various layers.
    - layers (list of dicts): Contains the configuration for the various layers.
    - data (numpy.ndarray): Input data.
    - labels (numpy.ndarray): Ground truth labels.

    Returns:
    - cp (dict): Contains cost and percentage values.
    - param_grad (list of dicts, optional): Contains gradient values for weights and biases.
    """

    l = len(layers)
    batch_size = layers[0]['batch_size']

    # Forward pass
    output = convnet_forward(params, layers, data)

    # Loss layer
    i = l - 1
    assert layers[i]['type'] == 'LOSS', 'last layer must be loss layer'
    wb = np.concatenate([params[i-1]['w'].ravel(), params[i-1]['b'].ravel()])
    cost, grad, input_od, percent = mlrloss(wb, output[i-1]['data'], labels, layers[i]['num'], 0, 1)

    cp = {'cost': cost / batch_size, 'percent': percent}
    
    # Backward pass
    param_grad = []
    if test is False:
        pg = {}
        pg['w'] = np.reshape(grad[:params[i-1]['w'].size], params[i-1]['w'].shape) / batch_size
        pg['b'] = np.reshape(grad[-params[i-1]['b'].size:], params[i-1]['b'].shape) / batch_size
        param_grad.append(pg)

        for i in range(l-2, 0, -1):
            layer_type = layers[i]['type']
            output[i]['diff'] = input_od

            pg = {}

            if layer_type == 'CONV':
                pg, input_od = conv_layer_backward(output[i], output[i-1], layers[i], params[i-1])
            elif layer_type == 'POOLING':
                input_od = pooling_layer_backward(output[i], output[i-1], layers[i])
                pg['w'] = []
                pg['b'] = []
            elif layer_type == 'IP':
                pg, input_od = inner_product_backward(output[i], output[i-1], layers[i], params[i-1])
            elif layer_type in 'RELU':
                input_od = relu_backward(output[i], output[i-1], layers[i])
                pg['w'] = []
                pg['b'] = []

            pg['w'] = np.array(pg['w']) / batch_size
            pg['b'] = np.array(pg['b']) / batch_size
            param_grad.append(pg)
    
    # reverse the param_grad list
    param_grad = param_grad[::-1]

    return cp, param_grad

def convnet_forward(params, layers, data, test=False):
    """
    Forward pass for a convolutional neural network.

    Parameters:
    - params (list of dicts): Contains the weights and biases for the various layers.
    - layers (list of dicts): Contains the configuration for the various layers.
    - data (numpy.ndarray): Input data.

    Returns:
    - output (list of dicts): Contains the output and other metadata for each layer.
    - P (numpy.ndarray, optional): Probabilities.
    """

    l = len(layers)
    assert layers[0]['type'] == 'DATA', 'first layer must be data layer'

    output = [{}]
    output[0]['data'] = data
    output[0]['height'] = layers[0]['height']
    output[0]['width'] = layers[0]['width']
    output[0]['channel'] = layers[0]['channel']
    output[0]['batch_size'] = layers[0]['batch_size']
    output[0]['diff'] = 0

    for i in range(1, l-1):
        layer_type = layers[i]['type']
        if layer_type == 'CONV':
            output.append(conv_layer_forward(output[i-1], layers[i], params[i-1]))
        elif layer_type == 'POOLING':
            output.append(pooling_layer_forward(output[i-1], layers[i]))
        elif layer_type == 'IP':
            output.append(inner_product_forward(output[i-1], layers[i], params[i-1]))
        elif layer_type == 'RELU':
            output.append(relu_forward(output[i-1]))
        else:
            raise Exception('Invalid layer type: %s' % layer_type)

    if test:
        W = params[l-2]['w'] @ output[l-2]['data'] + params[l-2]['b']
        W = np.vstack((W, np.zeros((1, W.shape[1]))))
        W -= np.max(W, axis=0)
        W = np.exp(W)

        # Convert to Probabilities by normalizing
        P = W / np.sum(W, axis=0)
        return output, P

    return output

