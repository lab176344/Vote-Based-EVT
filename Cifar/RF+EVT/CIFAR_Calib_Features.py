from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import load_model
import scipy.io as sio
from keras import backend as K
import numpy as np
import random



def normalize(X_train,X_test):
    #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test
# -----------------------------------------------------------------------------------------------------------------------
# Variable Initializations
num_classes = 6
m_Itotalfeatures=4500
m_startFeatures=4000
# input image dimensions
img_rows, img_cols = 32, 32
# -----------------------------------------------------------------------------------------------------------------------
# Load the data
# ----------------------------------------------------------------------------------------------------------------------
# Load the model
for x in range(5):
    random.seed(x)
    label = random.sample(range(10), 10)
    str1 = ''.join(str(e) for e in label)
    saveNameModel = str1 + 'CIFAR' + '.h5'
    model=load_model(saveNameModel)
    print(model.summary())
    #-------------------------------------------------------------------------------------------------------
    # Function to get a layer output
    get_1st_layer_output = K.function([model.layers[0].input],
                                      [model.layers[51].output])
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
        

    a1=[y_train == label[0]]
    a11 = np.reshape(a1[0], [a1[0].shape[0]])
    x_train_zeros = x_train[a11]
    layer_output_zeros = get_1st_layer_output([x_train_zeros])[0]
    layer_output=layer_output_zeros[m_startFeatures:m_Itotalfeatures]
    print('1 shape:', layer_output.shape)
    # Class 2
    a2=[y_train == label[1]]
    a21 = np.reshape(a2[0], [a2[0].shape[0]])
    x_train_ones = x_train[a21]
    layer_ouput_ones=get_1st_layer_output([x_train_ones])[0]
    layer_output=np.concatenate((layer_output,layer_ouput_ones[m_startFeatures:m_Itotalfeatures]),axis=0)
    print('2 shape:', layer_output.shape)
    # Class 3
    a3=[y_train == label[2]]
    a31 = np.reshape(a3[0], [a3[0].shape[0]])
    x_train_twos= x_train[a31]
    layer_ouput_twos=get_1st_layer_output([x_train_twos])[0]
    layer_output=np.concatenate((layer_output,layer_ouput_twos[m_startFeatures:m_Itotalfeatures]),axis=0)
    print('3 shape:', layer_output.shape)
    # Class 4
    a4=[y_train == label[3]]
    a41 = np.reshape(a4[0], [a4[0].shape[0]])
    x_train_threes= x_train[a41]
    layer_ouput_threes=get_1st_layer_output([x_train_threes])[0]
    layer_output=np.concatenate((layer_output,layer_ouput_threes[m_startFeatures:m_Itotalfeatures]),axis=0)
    print('4 shape:', layer_output.shape)
    # Class 5
    a5=[y_train == label[4]]
    a51 = np.reshape(a5[0], [a5[0].shape[0]])
    x_train_four= x_train[a51]
    layer_ouput_fours=get_1st_layer_output([x_train_four])[0]
    layer_output=np.concatenate((layer_output,layer_ouput_fours[m_startFeatures:m_Itotalfeatures]),axis=0)
    print('5 shape:', layer_output.shape)
    # Class 6
    a6=[y_train == label[5]]
    a61 = np.reshape(a6[0], [a6[0].shape[0]])
    x_train_five= x_train[a61]
    layer_ouput_fives=get_1st_layer_output([x_train_five])[0]
    layer_output=np.concatenate((layer_output,layer_ouput_fives[m_startFeatures:m_Itotalfeatures]),axis=0)
    print('6 shape:', layer_output.shape)

    saveNameFile = 'Calib\\'+ str1 + 'CiFar' + 'Calib.mat'
    output = {}
    output['CalibDataSet'] = layer_output
    output['CalibTarget'] = y_train_sorted
    sio.savemat(saveNameFile, output)

