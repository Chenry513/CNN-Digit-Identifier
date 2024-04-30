import numpy as np

def relu_forward(input_data):
    output = {
        'height': input_data['height'],
        'width': input_data['width'],
        'channel': input_data['channel'],
        'batch_size': input_data['batch_size'],
    }

    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    output['data'] = np.zeros_like(input_data['data'])
    output['data'] = np.maximum(0,input_data['data'])
        
    return output

def relu_backward(output, input_data, layer):
    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    input_od = np.zeros_like(input_data['data'])
    input_od[input_data['data'] >= 0] = 1 
    input_od = input_od * output['diff']

    return input_od
