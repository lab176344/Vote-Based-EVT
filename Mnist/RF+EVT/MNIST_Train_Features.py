from __future__ import print_function
import keras
from keras.datasets import mnist,fashion_mnist
from keras.models import load_model
from keras import backend as K
import scipy.io as sio
import os
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------------------------------------------------
# Variable Initializations

m_Itotalfeatures=5000
img_rows, img_cols = 28, 28

# input image dimensions

# -----------------------------------------------------------------------------------------------------------------------
# Load the data
for x in range(0, 5):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    test_set=np.zeros((x_train.shape[0], img_rows, img_cols, 1))
    for h in range(x_train.shape[0]):
        noise=np.random.uniform(low=0.0, high=1.0, size=(img_rows,img_cols,1))
        test_set[h]=noise
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    random.seed(x)
    label=random.sample(range(10),10)
    str1 = ''.join(str(e) for e in label[0:9])
    strsave=str1+'Mnist'+'.h5'
    saveNameModel=strsave
    model=load_model(saveNameModel)
    # -----------------------------------------------------------------------------------------------------------------------
    # Function to get a layer output
    get_1st_layer_output = K.function([model.layers[0].input],
                                  [model.layers[6].output])
    
    # for generating the noise data
    # -----------------------------------------------------------------------------------------------------------------------
    # Training data generate
    # Sort input
    idx_sorted=np.argsort(y_test)
    x_train_sorted=x_test[idx_sorted]
    y_train_sorted=y_test[idx_sorted]
    # ------------------------------------------------------------------------------------------------------------------
    # Class 0
    x_train_zeros = x_test[y_test == label[0]]
    layer_output_zeros = get_1st_layer_output([x_train_zeros])[0]
    layer_output=layer_output_zeros[0:m_Itotalfeatures]
    x_trainout=x_train_zeros[0:m_Itotalfeatures]
    print('1 shape:', layer_output_zeros.shape)
    # Class 1
    x_train_ones = x_test[y_test == label[1]]
    layer_ouput_ones=get_1st_layer_output([x_train_ones])[0]
    layer_output=np.concatenate((layer_output,layer_ouput_ones[0:m_Itotalfeatures]),axis=0)
    x_trainout=np.concatenate((x_trainout,x_train_ones[0:m_Itotalfeatures]),axis=0)
    # Class 2
    print('2 shape:', layer_ouput_ones.shape)
    x_train_twos= x_test[y_test == label[2]]
    layer_ouput_twos=get_1st_layer_output([x_train_twos])[0]
    layer_output=np.concatenate((layer_output,layer_ouput_twos[0:m_Itotalfeatures]),axis=0)
    x_trainout=np.concatenate((x_trainout,x_train_twos[0:m_Itotalfeatures]),axis=0)
    print('3 shape:', layer_ouput_twos.shape)
    # Class 3
    x_train_threes= x_test[y_test == label[3]]
    layer_ouput_threes=get_1st_layer_output([x_train_threes])[0]
    layer_output=np.concatenate((layer_output,layer_ouput_threes[0:m_Itotalfeatures]),axis=0)
    x_trainout=np.concatenate((x_trainout,x_train_threes[0:m_Itotalfeatures]),axis=0)
    # Class 4
    print('4 shape:', layer_ouput_threes.shape)
    x_train_fours= x_test[y_test == label[4]]
    layer_ouput_fours=get_1st_layer_output([x_train_fours])[0]
    layer_output=np.concatenate((layer_output,layer_ouput_fours[0:m_Itotalfeatures]),axis=0)
    x_trainout=np.concatenate((x_trainout,x_train_fours[0:m_Itotalfeatures]),axis=0)
    print('5 shape:', layer_ouput_fours.shape)
        # Class 5
    x_train_fives= x_test[y_test == label[5]]
    layer_ouput_fives=get_1st_layer_output([x_train_fives])[0]
    layer_output=np.concatenate((layer_output,layer_ouput_fives[0:m_Itotalfeatures]),axis=0)
    x_trainout=np.concatenate((x_trainout,x_train_fives[0:m_Itotalfeatures]),axis=0)
    print('6 shape:', layer_ouput_fives.shape)
    

    # Train target sort
    y_train_zeros=y_test[y_test == label[0]]
    y_train_sorted=np.zeros((np.size(y_train_zeros),),dtype=int)
    y_train_ones=y_test[y_test == label[1]]
    y_train_sorted=np.concatenate((y_train_sorted,np.ones((np.size(y_train_ones),),dtype=int)),axis=0)
    y_train_twos=y_test[y_test == label[2]]
    y_train_sorted=np.concatenate((y_train_sorted,2*np.ones((np.size(y_train_twos),),dtype=int)),axis=0)
    y_train_threes=y_test[y_test == label[3]]
    y_train_sorted=np.concatenate((y_train_sorted,3*np.ones((np.size(y_train_threes),),dtype=int)),axis=0)
    y_train_fours=y_test[y_test == label[4]]
    y_train_sorted=np.concatenate((y_train_sorted,4*np.ones((np.size(y_train_fours),),dtype=int)),axis=0)
    y_train_fives = y_test[y_test == label[5]]
    y_train_sorted = np.concatenate((y_train_sorted, 5*np.ones((np.size(y_train_fives),),dtype=int)), axis=0)
    
    print("Loaded dataset")
    print(model.summary())
    output = {}
    output['TrainDataSet'] = layer_output
    output['TrainTarget'] = y_train_sorted
    saveName='Train\TrainData'+str(x)+'.mat'
    sio.savemat(saveName, output)


    print(model.metrics_names)
    print("Loaded dataset")
    print(model.summary())