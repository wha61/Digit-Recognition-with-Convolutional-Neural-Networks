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
    output['data'] = np.maximum(input_data['data'], 0)
    
    # this part for testing in test_relu.py
    print("input: \n", input_data['data'])
    print("output: \n", output['data'])
    
    return output

def relu_backward(output, input_data, layer):
    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    input_od = np.zeros_like(input_data['data'])
    
    # gradient from last layer
    relu_diff = output["diff"]

    # derivative for input data
    input_od = input_data['data'] > 0

    input_od = relu_diff * input_od

    return input_od
