import numpy as np
from utils import im2col_conv_batch

def pooling_layer_forward(input, layer):
    """
    Forward pass for the pooling layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    """
    
    #  k represent the kernel size, s represent the stride and p represent the padding.
    h_in = input['height']
    w_in = input['width']
    c = input['channel']
    batch_size = input['batch_size']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']

    h_out = int((h_in + 2 * pad - k) / stride + 1)
    w_out = int((w_in + 2 * pad - k) / stride + 1)
    # print("data:", h_in, w_in, k, pad, stride, batch_size, c, h_out, w_out)
    
    output = {}
    output['height'] = h_out
    output['width'] = w_out
    output['channel'] = c
    output['batch_size'] = batch_size
    output['data'] = np.zeros((h_out *w_out *c, batch_size)) # replace with your implementation

    ###### Fill in the code here ######
    temp = np.zeros((h_out, w_out, c, batch_size))
    # Reshape input data for pooling operation
    input_data_reshaped = input['data'].reshape((h_in, w_in, c, batch_size), order = "F")
    # input_data_reshaped1= input['data'].reshape((h_in, w_in, c, batch_size))
    # print("input data:\n", input_data_reshaped1)
    # print("reshaped:\n ", input_data_reshaped)

    # Apply padding if necessary
    if pad > 0:
        input_data_reshaped = np.pad(input_data_reshaped, ((pad, pad), (pad, pad), (0, 0), (0, 0)), mode='constant')
    
    for i in range(batch_size):
        for ch in range(c): 
            for h in range(h_out): # use h_out
                for w in range(w_out): # use  w_out                      
                    pool_patch = input_data_reshaped[h*stride: h*stride + k, w*stride: w*stride + k, ch, i]
                    temp[h, w, ch, i] = np.max(pool_patch)
    # Reshape the output data to [height × width × channel, batch size]
    output['data'] = temp.reshape((h_out * w_out * c, batch_size), order = "F")
    
    return output

def pooling_layer_backward(output, input, layer):
    """
    Backward pass for the pooling layer.

    Parameters:
    - output (dict): Contains the gradients from the next layer.
    - input (dict): Contains the original input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.

    Returns:
    - input_od (numpy.ndarray): Gradient with respect to the input.
    """

    h_in = input['height']
    w_in = input['width']
    c = input['channel']
    batch_size = input['batch_size']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']

    h_out = (h_in + 2*pad - k) // stride + 1
    w_out = (w_in + 2*pad - k) // stride + 1

    input_od = np.zeros(input['data'].shape)
    input_od = input_od.reshape(h_in * w_in * c * batch_size, 1)

    im_b = np.reshape(input['data'], (h_in, w_in, c, batch_size), order='F')
    im_b = np.pad(im_b, ((pad, pad), (pad, pad), (0, 0), (0, 0)), mode='constant')
    
    diff = np.reshape(output['diff'], (h_out*w_out, c*batch_size), order='F')

    for h in range(h_out):
        for w in range(w_out):
            matrix_hw = im_b[h*stride : h*stride + k, w*stride : w*stride + k, :, :]
            flat_matrix = matrix_hw.reshape((k*k, c*batch_size), order='F')
            i1 = np.argmax(flat_matrix, axis=0)
            R, C = np.unravel_index(i1, matrix_hw.shape[:2], order='F')
            nR = h*stride + R
            nC = w*stride + C
            i2 = np.ravel_multi_index((nR, nC), (h_in, w_in), order='F')
            i4 = np.ravel_multi_index((i2, np.arange(c*batch_size)), (h_in*w_in, c*batch_size), order='F')
            i3 = np.ravel_multi_index((h, w), (h_out, w_out), order='F')
            input_od[i4] += diff[i3:i3+1, :].T

    input_od = np.reshape(input_od, (h_in*w_in, c*batch_size), order='F')
    input_od = np.reshape(input_od, (h_in*w_in*c, batch_size), order='F')

    return input_od
