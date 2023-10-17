import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.
    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    # data - stores the actual data being passed between the layers. 
    # This is always supposed to be of the size [ height × width × channel, batch size ]. 
    # You can resize this structure during computations, but make sure to revert it to a two-dimensional matrix.
    # The data is stored in a column major order.
    # The row comes next, and the channel comes the last.
    # d is height × width × channel, k is batch size 
    d, k = input["data"].shape
    # n is the dimensionality of the previous layer
    n = param["w"].shape[1]
    ###### Fill in the code here ######
    # f (x) = W x + b
    # x [d * k]
    # W [m * n]
    # b [m * 1]
    # so n = d here
    output_data = np.dot(param["w"].T, input["data"]) + param["b"].reshape(-1, 1)
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

    # dY  dl/dy
    dY = output["diff"]
    # xT
    input_transp = input_data['data'].T
    # dw dl/dw = xT * dY
    param_grad['w'] = np.dot(input_data['data'], dY.T)
    param_grad['b'] = np.sum(dY.T, axis=0)
    # dl/dhi-1 : dx = dY * wT 
    input_od = np.dot(param["w"], dY)

    return param_grad, input_od