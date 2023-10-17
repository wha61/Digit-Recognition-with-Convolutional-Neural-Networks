import numpy as np
import matplotlib.pyplot as plt
from relu import relu_forward

def test_relu():
    input_data = {'data': np.zeros((3*3,1))}
    input_data['data'][1, 0] = 0.5
    input_data['data'][2, 0] = 0.25
    input_data['data'][6, 0]= -0.5

    input_data['width'] = 3
    input_data['height'] = 3
    input_data['channel'] = 1
    input_data['batch_size'] = 1

    relu_forward(input_data)

test_relu()



