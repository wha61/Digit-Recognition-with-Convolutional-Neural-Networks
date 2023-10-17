import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image

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

def test_single_image(img_path, params, layers):
    # open image
    img = Image.open(img_path)

    # Convert the image to grayscale
    img = img.convert('L')

    # Resize the image to 28x28
    img = img.resize((28, 28))

    # revert color to adapt minist data 
    img = Image.fromarray(255 - np.array(img))

    plt.imshow(img, cmap='gray')
    plt.title('My Image')
    plt.axis('off')
    plt.show()

    # Convert image to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Reshape the image to match the expected input shape for the model
    # (batch_size, height, width, channel)
    img_array = np.reshape(img_array, (28, 28), order='F')
    # img_array = img_array.reshape(1, 28, 28, 1)
    
    # Update the batch_size for the network
    layers[0]['batch_size'] = 1

    # Forward pass
    _, P = convnet_forward(params, layers, img_array, test=True)

    # Get the predicted class
    predicted_class = P.argmax(axis = 0)

    return predicted_class[0]

# Test the function
img_path = '../images/3.jpg'
predicted_class = test_single_image(img_path, params, layers)
i = Image.open(img_path)
plt.imshow(i, cmap='gray')
plt.title(f"Predicted: {predicted_class}")
plt.axis('off') 
plt.show()
print(f"The predicted class for the image is: {predicted_class}")
