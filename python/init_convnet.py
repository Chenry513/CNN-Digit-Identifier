import numpy as np

def init_convnet(layers):
    h = layers[0]['height']
    w = layers[0]['width']
    c = layers[0]['channel']
    
    params = [None] * (len(layers) - 1)
    
    for i in range(1, len(layers)):
        layer_type = layers[i]['type']
        
        if layer_type == 'CONV':
            scale = np.sqrt(3 / (h * w * c))
            params[i-1] = {
                'w': 2 * scale * np.random.rand(layers[i]['k']*layers[i]['k']*c//layers[i]['group'], layers[i]['num']) - scale,
                'b': np.zeros((1, layers[i]['num']))
            }
            h = (h + 2*layers[i]['pad'] - layers[i]['k']) // layers[i]['stride'] + 1
            w = (w + 2*layers[i]['pad'] - layers[i]['k']) // layers[i]['stride'] + 1
            c = layers[i]['num']
            
        elif layer_type == 'POOLING':
            h = (h - layers[i]['k']) // layers[i]['stride'] + 1
            w = (w - layers[i]['k']) // layers[i]['stride'] + 1
            params[i-1] = {'w': np.zeros((0,0)), 'b': np.zeros((0,0))}
            
        elif layer_type == 'IP':
            scale = np.sqrt(3 / (h * w * c))
            if layers[i]['init_type'] == 'gaussian':
                params[i-1] = {
                    'w': scale * np.random.randn(h*w*c, layers[i]['num']),
                    'b': np.zeros((1, layers[i]['num']))
                }
            elif layers[i]['init_type'] == 'uniform':
                params[i-1] = {
                    'w': 2 * scale * np.random.rand(h*w*c, layers[i]['num']) - scale,
                    'b': np.zeros((1, layers[i]['num']))
                }
            h, w, c = 1, 1, layers[i]['num']
            
        elif layer_type in ['RELU', 'ELU']:
            params[i-1] = {'w': np.zeros((0,0)), 'b': np.zeros((0,0))}
            
        elif layer_type == 'LOSS':
            scale = np.sqrt(3 / (h * w * c))
            params[i-1] = {
                'w': (2 * scale * np.random.rand(h*w*c, layers[i]['num'] - 1) - scale).T,
                'b': np.zeros((1, layers[i]['num'] - 1)).T
            }
            h, w, c = 1, 1, layers[i]['num']
    
    return params

