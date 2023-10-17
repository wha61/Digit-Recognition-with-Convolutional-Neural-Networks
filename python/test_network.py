import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


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

# print("test:", xtest.shape, ytest.shape)
# print("xtest:", xtest[:, 150].shape)


# Testing the network
#### Modify the code to get the confusion matrix ####
all_preds = []
trues = []
# i = 0, 100, 200, 300, 400
for i in range(0, xtest.shape[1], 100):
    cptest, P = convnet_forward(params, layers, xtest[:,i:i+100], test=True)
    pMax = P.argmax(axis = 0)
    # print("P", P.shape)
    # print("pMax", pMax.shape)
    all_preds.extend(pMax)
    trues.append(ytest[:,i:i+100])

# print("preds", all_preds)
# print("trues", trues)
flat_trues = [item for sublist in trues for item in sublist[0]]
# print("flat_trues", flat_trues)

con = confusion_matrix(flat_trues, all_preds)
# print(con)
# hint: 
#     you can use confusion_matrix from sklearn.metrics (pip install -U scikit-learn)
#     to compute the confusion matrix. Or you can write your own code :)

plt.imshow(con, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))
plt.ylabel('True label')
plt.xlabel('Predicted label')

for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    plt.text(j, i, con[i, j],
             horizontalalignment="center",
             color="white" if con[i, j] > con.max() / 2 else "black")

plt.show()