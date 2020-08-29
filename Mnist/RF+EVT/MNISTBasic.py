from __future__ import print_function
import keras
from keras.datasets import mnist,fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import random 

# input image dimensions
batch_size = 128
num_classes = 6
epochs = 20
m_Itotalfeatures=4000
img_rows, img_cols = 28, 28


# sorted mnist dataset training
for x in range(0, 5):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
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
    saveName=strsave
    #train input sort
    x_train_zeros = x_train[y_train == label[0]]
    x_train_sorted=x_train_zeros[0:m_Itotalfeatures]
    x_train_ones = x_train[y_train == label[1]]
    x_train_sorted=np.concatenate((x_train_sorted,x_train_ones[0:m_Itotalfeatures]),axis=0)
    x_train_twos= x_train[y_train == label[2]]
    x_train_sorted=np.concatenate((x_train_sorted,x_train_twos[0:m_Itotalfeatures]),axis=0)
    x_train_threes= x_train[y_train == label[3]]
    x_train_sorted=np.concatenate((x_train_sorted,x_train_threes[0:m_Itotalfeatures]),axis=0)
    x_train_four= x_train[y_train == label[4]]
    x_train_sorted=np.concatenate((x_train_sorted,x_train_four[0:m_Itotalfeatures]),axis=0)
    x_train_five= x_train[y_train == label[5]]
    x_train_sorted=np.concatenate((x_train_sorted,x_train_five[0:m_Itotalfeatures]),axis=0)

    
    #train target sort
    y_train_zeros=y_train[y_train ==label[0]]
    y_train_sorted=np.zeros((m_Itotalfeatures,), dtype=int)
    y_train_ones=y_train[y_train ==label[1]]
    y_train_sorted=np.concatenate((y_train_sorted,1*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
    y_train_twos=y_train[y_train == label[2]]
    y_train_sorted=np.concatenate((y_train_sorted,2*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
    y_train_threes=y_train[y_train == label[3]]
    y_train_sorted=np.concatenate((y_train_sorted,3*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
    y_train_four=y_train[y_train == label[4]]
    y_train_sorted=np.concatenate((y_train_sorted,4*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
    y_train_five=y_train[y_train == label[5]]
    y_train_sorted=np.concatenate((y_train_sorted,5*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)


    ##sorted mnist dataset testing
    idx_sorted_test=np.argsort(y_test)
    x_test_zeros = x_test[y_test == label[0]]
    x_test_sorted=x_test_zeros
    x_test_ones = x_test[y_test == label[1]]
    x_test_sorted=np.concatenate((x_test_sorted,x_test_ones),axis=0)
    x_test_twos= x_test[y_test == label[2]]
    x_test_sorted=np.concatenate((x_test_sorted,x_test_twos),axis=0)
    x_test_threes= x_test[y_test ==  label[3]]
    x_test_sorted=np.concatenate((x_test_sorted,x_test_threes),axis=0)
    x_test_four= x_test[y_test ==  label[4]]
    x_test_sorted=np.concatenate((x_test_sorted,x_test_four),axis=0)
    x_test_five= x_test[y_test ==  label[5]]
    x_test_sorted=np.concatenate((x_test_sorted,x_test_five),axis=0)

    ##test target sort
    y_test_zeros=y_test[y_test ==  label[0]]
    y_test_sorted=np.zeros((np.size(y_test_zeros),), dtype=int)
    y_test_ones=y_test[y_test ==  label[1]]
    y_test_sorted=np.concatenate((y_test_sorted,1*np.ones((np.size(y_test_ones),), dtype=int)),axis=0)
    y_test_twos=y_test[y_test ==  label[2]]
    y_test_sorted=np.concatenate((y_test_sorted,2*np.ones((np.size(y_test_twos),), dtype=int)),axis=0)
    y_test_threes=y_test[y_test ==  label[3]]
    y_test_sorted=np.concatenate((y_test_sorted,3*np.ones((np.size(y_test_threes),), dtype=int)),axis=0)
    y_test_four=y_test[y_test ==  label[4]]
    y_test_sorted=np.concatenate((y_test_sorted,4*np.ones((np.size(y_test_four),), dtype=int)),axis=0)
    y_test_five=y_test[y_test ==  label[5]]
    y_test_sorted=np.concatenate((y_test_sorted,5*np.ones((np.size(y_test_five),), dtype=int)),axis=0)
  
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train_sorted, num_classes)
    y_test = keras.utils.to_categorical(y_test_sorted, num_classes)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = 'softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    hist = model.fit(x_train_sorted, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test_sorted, y_test))
    score = model.evaluate(x_test_sorted, y_test, verbose=0)
    model.save(saveName)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
