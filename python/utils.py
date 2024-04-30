import numpy as np
import copy

def col2im_conv(col, input_data, layer, h_out, w_out):
    """
    Convert column representation to image representation for convolution.
    
    Parameters:
    - col (numpy.ndarray): The column representation.
    - input_data (dict): A dictionary containing the original input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    - h_out (int): Height of the output after convolution.
    - w_out (int): Width of the output after convolution.

    Returns:
    - im (numpy.ndarray): The image representation.
    """
    
    h_in = input_data['height']
    w_in = input_data['width']
    c = input_data['channel']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']

    im = np.zeros((h_in, w_in, c))
    
    assert pad == 0, 'pad must be 0'

    col = np.reshape(col, (k*k*c, h_out*w_out), order='F')
    
    for h in range(1, h_out+1):
        for w in range(1, w_out+1):
            im_slice = im[(h-1)*stride : (h-1)*stride + k, (w-1)*stride : (w-1)*stride + k, :]
            col_slice = col[:, h-1 + (w-1)*h_out]
            col_slice_reshaped = np.reshape(col_slice, (k, k, c), order='F')
            im[(h-1)*stride : (h-1)*stride + k, (w-1)*stride : (w-1)*stride + k, :] = im_slice + col_slice_reshaped
    
    im = im[pad:-(pad+1), pad:-(pad+1), :] if pad > 0 else im

    return im

def im2col_conv(input_n, layer, h_out, w_out):
    """
    Convert image representation to column representation for convolution.
    
    Parameters:
    - input_n (dict): A dictionary containing the input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    - h_out (int): Height of the output after convolution.
    - w_out (int): Width of the output after convolution.

    Returns:
    - col (numpy.ndarray): The column representation.
    """

    h_in = input_n['height']
    w_in = input_n['width']
    c = input_n['channel']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']

    im = np.reshape(input_n['data'], (h_in, w_in, c), order='F')
    
    if pad > 0:
        im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    col = np.zeros((k*k*c, h_out*w_out))

    for h in range(h_out):
        for w in range(w_out):
            matrix_hw = im[h*stride : h*stride + k, w*stride : w*stride + k, :]
            col[:, h + w*h_out] = matrix_hw.ravel(order='F')

    col = col.ravel(order='F')

    return col

def im2col_conv_batch(input_n, layer, h_out, w_out):
    batch_size = input_n['batch_size']
    h_in = input_n['height']
    w_in = input_n['width']
    c = input_n['channel']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']

    im = input_n['data'].reshape((c, h_in, w_in, batch_size))
    im = np.transpose(im, (1, 2, 0, 3))
    im = np.pad(im, ((pad, pad), (pad, pad), (0, 0), (0, 0)), mode='constant')
    
    col = np.zeros((k*k*c, h_out*w_out, batch_size))

    for h in range(h_out):
        for w in range(w_out):
            matrix_hw = im[h*stride : h*stride + k, w*stride : w*stride + k, :, :]
            matrix_hw = matrix_hw.transpose((2, 0, 1, 3))
            col[:, h*h_out + w, :] = matrix_hw.reshape((-1, batch_size))
    
    return col

def sgd_momentum(rate, mu, weight_decay, params, param_winc, param_grad):
    """
    Update the parameter with SGD with momentum.

    :param rate: Learning rate at current step
    :param mu: Momentum
    :param weight_decay: Weight decay
    :param params: Original weight parameters
    :param param_winc: Buffer to store history gradient accumulation
    :param param_grad: Gradient of parameter

    :return: Updated parameters and buffer
    """
    params = copy.deepcopy(params)
    param_winc = copy.deepcopy(param_winc)
    for l_idx in range(len(params)):
        param_winc[l_idx]['w'] = mu * param_winc[l_idx]['w'] + rate * (param_grad[l_idx]['w'] + weight_decay * params[l_idx]['w'])
        param_winc[l_idx]['b'] = mu * param_winc[l_idx]['b'] + rate * (param_grad[l_idx]['b'])
        params[l_idx]['w'] = params[l_idx]['w'] - param_winc[l_idx]['w']
        params[l_idx]['b'] = params[l_idx]['b'] - param_winc[l_idx]['b']

    return params, param_winc 

def get_lr(iter, epsilon, gamma, power):
    """
    Get the learning rate at step iter
    """
    lr_t = epsilon / (1 + gamma * iter) ** power
    return lr_t

def get_lenet(batch_size=100):
    layers = []

    # Layer 1: DATA
    layers.append({
        'type': 'DATA',
        'height': 28,
        'width': 28,
        'channel': 1,
        'batch_size': batch_size
    })

    # Layer 2: CONV
    layers.append({
        'type': 'CONV',
        'num': 20,
        'k': 5,
        'stride': 1,
        'pad': 0,
        'group': 1
    })

    # Layer 3: RELU
    layers.append({
        'type': 'RELU'
    })

    # Layer 4: POOLING
    layers.append({
        'type': 'POOLING',
        'k': 2,
        'stride': 2,
        'pad': 0
    })

    # Layer 5: CONV
    layers.append({
        'type': 'CONV',
        'k': 5,
        'stride': 1,
        'pad': 0,
        'group': 1,
        'num': 50
    })

    # Layer 6: RELU
    layers.append({
        'type': 'RELU'
    })

    # Layer 7: POOLING
    layers.append({
        'type': 'POOLING',
        'k': 2,
        'stride': 2,
        'pad': 0
    })

    # Layer 8: IP
    layers.append({
        'type': 'IP',
        'num': 500,
        'init_type': 'uniform'
    })

    # Layer 9: RELU
    layers.append({
        'type': 'RELU'
    })

    # Layer 10: LOSS
    layers.append({
        'type': 'LOSS',
        'num': 10
    })

    return layers

