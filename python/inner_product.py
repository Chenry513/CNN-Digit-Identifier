import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    d, k = input["data"].shape
    n = param["w"].shape[1]

    ###### Fill in the code here ######
    output_data = np.dot(param["w"].T, input["data"]) + param["b"].reshape((n, 1))

    # Initialize output data structure
    
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": output_data # replace 'data' value with your implementation
    }
    
    

    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.
    param_grad['b'] = np.zeros_like(param['b'])
    param_grad['w'] = np.zeros_like(param['w'])
    input_od = None

    data = input_data['data']
    diff = output['diff']
    weights = param['w']

    param_grad = {}
    param_grad['b'] = np.sum(diff, axis=1)
    param_grad['w'] = np.dot(data, diff.T)
    input_od = np.dot(weights, diff)

    return param_grad, input_od