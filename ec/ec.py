import cv2
import numpy as np
from PIL import Image
import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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


def test_single_image_array(img_array, params, layers):
    # Convert numpy array to PIL Image
    img = Image.fromarray((img_array * 255).astype(np.uint8))

    # Convert the image to grayscale
    img = img.convert('L')

    # Resize the image to 28x28
    img = img.resize((28, 28))

    # plt.imshow(img, cmap='gray')
    # plt.title('My Image')
    # plt.axis('off')  # Turn off axis numbers and ticks
    # plt.show()

    # Convert image to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Reshape the image to match the expected input shape for the model
    # (batch_size, height, width, channel)
    img_array = np.reshape(img_array, (28, 28), order='F')

    # Update the batch_size for the network
    layers[0]['batch_size'] = 1

    # Forward pass
    _, P = convnet_forward(params, layers, img_array, test=True)

    # Get the predicted class
    predicted_class = P.argmax(axis=0)

    return predicted_class[0]


def classify_digit(img_path, params, layers, filter, padding, erode_iter, dilate_iter):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predicted_classes = []
    roi_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = binary[y:y+h, x:x+w]

        # to ingore some noise point cluster
        if cv2.contourArea(contour) < filter:
            continue
        
        # add pading to make img similar to train set
        top = padding
        bottom = padding
        left = padding
        right = padding

        padded_roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        # plt.imshow(padded_roi, cmap='gray')
        # plt.title("Padded ROI")
        # plt.axis('off') 
        # plt.show()

        # decrease noise point and bold number
        kernel = np.ones((2,2),np.uint8)
        padded_roi = cv2.erode(padded_roi,kernel,iterations = erode_iter)
        padded_roi = cv2.dilate(padded_roi,kernel,iterations = dilate_iter)

        resized_roi = cv2.resize(padded_roi, (28, 28), interpolation=cv2.INTER_AREA)
        resized_roi = resized_roi / 255.0  # Normalize

        # predict
        predicted_class = test_single_image_array(resized_roi, params, layers)
        # plt.imshow(resized_roi, cmap='gray')
        # plt.title(f"Predicted: {predicted_class}")
        # plt.axis('off') 
        # plt.show()
        predicted_classes.append(predicted_class)
        roi_images.append(resized_roi)

        # print(f"The predicted class for the component {contour} is: {predicted_class}")
    total_images = len(roi_images)
    plt.figure(figsize=(10, 10))

    for i in range(total_images):
        plt.subplot(10, 5, i+1)  # Adjust if you expect more than 25 images
        plt.imshow(roi_images[i], cmap='gray')
        plt.title(f"Predicted: {predicted_classes[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    return predicted_classes, roi_images



# change filter, padding, erode_iter, dilate_iter to optimize
classify_digit('../images/image1.jpg', params, layers, 500, 30, 2, 8)
classify_digit('../images/image2.jpg', params, layers, 500, 30, 2, 8)
classify_digit('../images/image3.png', params, layers, 500, 30, 2, 8)
classify_digit('../images/image4.jpg', params, layers, 40, 10, 1, 2)

