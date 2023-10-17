import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)
m_train = xtrain.shape[1]

batch_size = 1
layers[0]['batch_size'] = batch_size

img = xtest[:,0]
img = np.reshape(img, (28, 28), order='F')
plt.imshow(img.T, cmap='gray')
plt.show()

output = convnet_forward(params, layers, xtest[:,0:1])
output_1 = np.reshape(output[0]['data'], (28,28), order='F')

##### Fill in your code here to plot the features ######
output = convnet_forward(params, layers, xtest[:,0:1])

# Assuming the second layer (CONV layer) output is indexed by 1 and third layer (ReLU layer) by 2
conv_output = output[1]['data']  # Data from CONV layer
print(conv_output.shape)

relu_output = output[2]['data']  # Data from ReLU layer
print(relu_output.shape)

final_output = output[8]['data']

# calculate size of 1 in 20
num_features = 20 
s = int(np.sqrt(conv_output.shape[0]/num_features))

# conv out
plt.figure(figsize=(10, 8))
for i in range(num_features):
    plt.subplot(4, 5, i+1)
    feature_map = conv_output[i*s*s:(i+1)*s*s]
    plt.imshow(feature_map.reshape(s, s), cmap='gray')
    plt.axis('off')
plt.suptitle('CONV Layer Outputs')
plt.show()

num_features = 20 
s = int(np.sqrt(relu_output.shape[0]/num_features))

# relu out
plt.figure(figsize=(10, 8))
for i in range(num_features):
    plt.subplot(4, 5, i+1)
    feature_map = relu_output[i*s*s:(i+1)*s*s]
    plt.imshow(feature_map.reshape(s, s), cmap='gray')
    plt.axis('off')
plt.suptitle('ReLU Layer Outputs')
plt.show()

# final out, for testing
num_features = 20  
s = 5
plt.figure(figsize=(10, 8))
for i in range(num_features):
    plt.subplot(4, 5, i+1)
    feature_map = final_output[i*s*s:(i+1)*s*s]
    plt.imshow(feature_map.reshape(s, s), cmap='gray')
    plt.axis('off')
plt.suptitle('finals Outputs')
plt.show()

