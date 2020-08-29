
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import random

class cifar10vgg:
    def __init__(self,train=True):
        self.num_classes = 6
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):
        label=random.sample(range(10),10)
        str1 = ''.join(str(e) for e in label)
        strsave=str1+'CIFAR'+'.h5'
        saveName=strsave
        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)
        m_Itotalfeatures=4000
        # sorted mnist dataset training
        idx_sorted=np.argsort(y_train)
        a1=[y_train == label[0]]
        a11 = np.reshape(a1[0], [a1[0].shape[0]])
        x_train_zeros = x_train[a11]
        x_train_sorted=x_train_zeros[0:m_Itotalfeatures]
        a2=[y_train == label[1]]
        a21 = np.reshape(a2[0], [a2[0].shape[0]])
        x_train_ones = x_train[a21]
        x_train_sorted=np.concatenate((x_train_sorted,x_train_ones[0:m_Itotalfeatures]),axis=0)
        a3=[y_train == label[2]]
        a31 = np.reshape(a3[0], [a3[0].shape[0]])
        x_train_twos= x_train[a31]
        x_train_sorted=np.concatenate((x_train_sorted,x_train_twos[0:m_Itotalfeatures]),axis=0)
        a4=[y_train == label[3]]
        a41 = np.reshape(a4[0], [a4[0].shape[0]])
        x_train_threes= x_train[a41]
        x_train_sorted=np.concatenate((x_train_sorted,x_train_threes[0:m_Itotalfeatures]),axis=0)
        a5=[y_train == label[4]]
        a51 = np.reshape(a5[0], [a5[0].shape[0]])
        x_train_four= x_train[a51]
        x_train_sorted=np.concatenate((x_train_sorted,x_train_four[0:m_Itotalfeatures]),axis=0)
        a6=[y_train == label[5]]
        a61 = np.reshape(a6[0], [a6[0].shape[0]])
        x_train_five= x_train[a61]
        x_train_sorted=np.concatenate((x_train_sorted,x_train_five[0:m_Itotalfeatures]),axis=0)
      
        #train target sort
        y_train_zeros=y_train[y_train == label[0]]
        y_train_sorted=np.zeros(((m_Itotalfeatures),), dtype=int)
        y_train_ones=y_train[y_train == label[1]]
        y_train_sorted=np.concatenate((y_train_sorted,1*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
        y_train_twos=y_train[y_train == label[2]]
        y_train_sorted=np.concatenate((y_train_sorted,2*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
        y_train_threes=y_train[y_train == label[3]]
        y_train_sorted=np.concatenate((y_train_sorted,3*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
        y_train_four=y_train[y_train == label[4]]
        y_train_sorted=np.concatenate((y_train_sorted,4*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
        y_train_five=y_train[y_train == label[5]]
        y_train_sorted=np.concatenate((y_train_sorted,5*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)

        #sorted mnist dataset testing
        idx_sorted_test=np.argsort(y_test)
        b1=[y_test == label[0]]
        b11 = np.reshape(b1[0], [b1[0].shape[0]])
        x_test_zeros = x_test[b11]
        x_test_sorted=x_test_zeros
        b2=[y_test == label[1]]
        b21 = np.reshape(b2[0], [b2[0].shape[0]])
        x_test_ones = x_test[b21]
        x_test_sorted=np.concatenate((x_test_sorted,x_test_ones),axis=0)
        b3=[y_test == label[2]]
        b31 = np.reshape(b3[0], [b3[0].shape[0]])
        x_test_twos= x_test[b31]
        x_test_sorted=np.concatenate((x_test_sorted,x_test_twos),axis=0)
        b4=[y_test == label[3]]
        b41 = np.reshape(b4[0], [b4[0].shape[0]])
        x_test_threes= x_test[b41]
        x_test_sorted=np.concatenate((x_test_sorted,x_test_threes),axis=0)
        b5=[y_test == label[4]]
        b51 = np.reshape(b5[0], [b5[0].shape[0]])
        x_test_four= x_test[b51]
        x_test_sorted=np.concatenate((x_test_sorted,x_test_four),axis=0)
        b6=[y_test == label[5]]
        b61 = np.reshape(b6[0], [b6[0].shape[0]])
        x_test_five= x_test[b61]
        x_test_sorted=np.concatenate((x_test_sorted,x_test_five),axis=0)	
		
        #test target sort
        y_test_zeros=y_test[y_test == label[0]]
        y_test_sorted=np.zeros((np.size(y_test_zeros),), dtype=int)
        y_test_ones=y_test[y_test == label[1]]
        y_test_sorted=np.concatenate((y_test_sorted,1*np.ones((np.size(y_test_ones),), dtype=int)),axis=0)
        y_test_twos=y_test[y_test == label[2]]
        y_test_sorted=np.concatenate((y_test_sorted,2*np.ones((np.size(y_test_twos),), dtype=int)),axis=0)
        y_test_threes=y_test[y_test == label[3]]
        y_test_sorted=np.concatenate((y_test_sorted,3*np.ones((np.size(y_test_threes),), dtype=int)),axis=0)
        y_test_four=y_test[y_test == label[4]]
        y_test_sorted=np.concatenate((y_test_sorted,4*np.ones((np.size(y_test_four),), dtype=int)),axis=0)
        y_test_five=y_test[y_test == label[5]]
        y_test_sorted=np.concatenate((y_test_sorted,5*np.ones((np.size(y_test_five),), dtype=int)),axis=0)

        y_train = keras.utils.to_categorical(y_train_sorted, self.num_classes)
        y_test = keras.utils.to_categorical(y_test_sorted, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train_sorted)

        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit_generator(datagen.flow(x_train_sorted, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train_sorted.shape[0] // batch_size,
                            epochs=maxepoches,shuffle=True,
                            validation_data=(x_test_sorted, y_test),callbacks=[reduce_lr],verbose=2)
        model.save(saveName)
        return model

if __name__ == '__main__':

    for x in range(0,5):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        num_classes = 6
        m_Itotalfeatures=4000
        random.seed(x)
        label=random.sample(range(10),10)
        str1 = ''.join(str(e) for e in label)
        strsave=str1+'CIFAR'+'.h5'
        saveName=strsave
        label = random.sample(range(10), 10)
        # sorted mnist dataset training
        idx_sorted=np.argsort(y_train)
        a1=[y_train == label[0]]
        a11 = np.reshape(a1[0], [a1[0].shape[0]])
        x_train_zeros = x_train[a11]
        x_train_sorted=x_train_zeros[0:m_Itotalfeatures]
        a2=[y_train == label[1]]
        a21 = np.reshape(a2[0], [a2[0].shape[0]])
        x_train_ones = x_train[a21]
        x_train_sorted=np.concatenate((x_train_sorted,x_train_ones[0:m_Itotalfeatures]),axis=0)
        a3=[y_train == label[2]]
        a31 = np.reshape(a3[0], [a3[0].shape[0]])
        x_train_twos= x_train[a31]
        x_train_sorted=np.concatenate((x_train_sorted,x_train_twos[0:m_Itotalfeatures]),axis=0)
        a4=[y_train == label[3]]
        a41 = np.reshape(a4[0], [a4[0].shape[0]])
        x_train_threes= x_train[a41]
        x_train_sorted=np.concatenate((x_train_sorted,x_train_threes[0:m_Itotalfeatures]),axis=0)
        a5=[y_train == label[4]]
        a51 = np.reshape(a5[0], [a5[0].shape[0]])
        x_train_four= x_train[a51]
        x_train_sorted=np.concatenate((x_train_sorted,x_train_four[0:m_Itotalfeatures]),axis=0)
        a6=[y_train == label[5]]
        a61 = np.reshape(a6[0], [a6[0].shape[0]])
        x_train_five= x_train[a61]
        x_train_sorted=np.concatenate((x_train_sorted,x_train_five[0:m_Itotalfeatures]),axis=0)

        #train target sort
        y_train_zeros=y_train[y_train == label[0]]
        y_train_sorted=np.zeros(((m_Itotalfeatures),), dtype=int)
        y_train_ones=y_train[y_train == label[1]]
        y_train_sorted=np.concatenate((y_train_sorted,1*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
        y_train_twos=y_train[y_train == label[2]]
        y_train_sorted=np.concatenate((y_train_sorted,2*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
        y_train_threes=y_train[y_train == label[3]]
        y_train_sorted=np.concatenate((y_train_sorted,3*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
        y_train_four=y_train[y_train == label[4]]
        y_train_sorted=np.concatenate((y_train_sorted,4*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)
        y_train_five=y_train[y_train == label[5]]
        y_train_sorted=np.concatenate((y_train_sorted,5*np.ones((m_Itotalfeatures,),dtype=int)),axis=0)

        #sorted mnist dataset testing
        idx_sorted_test=np.argsort(y_test)
        b1=[y_test == label[0]]
        b11 = np.reshape(b1[0], [b1[0].shape[0]])
        x_test_zeros = x_test[b11]
        x_test_sorted=x_test_zeros
        b2=[y_test == label[1]]
        b21 = np.reshape(b2[0], [b2[0].shape[0]])
        x_test_ones = x_test[b21]
        x_test_sorted=np.concatenate((x_test_sorted,x_test_ones),axis=0)
        b3=[y_test == label[2]]
        b31 = np.reshape(b3[0], [b3[0].shape[0]])
        x_test_twos= x_test[b31]
        x_test_sorted=np.concatenate((x_test_sorted,x_test_twos),axis=0)
        b4=[y_test == label[3]]
        b41 = np.reshape(b4[0], [b4[0].shape[0]])
        x_test_threes= x_test[b41]
        x_test_sorted=np.concatenate((x_test_sorted,x_test_threes),axis=0)
        b5=[y_test == label[4]]
        b51 = np.reshape(b5[0], [b5[0].shape[0]])
        x_test_four= x_test[b51]
        x_test_sorted=np.concatenate((x_test_sorted,x_test_four),axis=0)
        b6=[y_test == label[5]]
        b61 = np.reshape(b6[0], [b6[0].shape[0]])
        x_test_five= x_test[b61]
        x_test_sorted=np.concatenate((x_test_sorted,x_test_five),axis=0)

        #test target sort
        y_test_zeros=y_test[y_test == label[0]]
        y_test_sorted=np.zeros((np.size(y_test_zeros),), dtype=int)
        y_test_ones=y_test[y_test == label[1]]
        y_test_sorted=np.concatenate((y_test_sorted,1*np.ones((np.size(y_test_ones),), dtype=int)),axis=0)
        y_test_twos=y_test[y_test == label[2]]
        y_test_sorted=np.concatenate((y_test_sorted,2*np.ones((np.size(y_test_twos),), dtype=int)),axis=0)
        y_test_threes=y_test[y_test == label[3]]
        y_test_sorted=np.concatenate((y_test_sorted,3*np.ones((np.size(y_test_threes),), dtype=int)),axis=0)
        y_test_four=y_test[y_test == label[4]]
        y_test_sorted=np.concatenate((y_test_sorted,4*np.ones((np.size(y_test_four),), dtype=int)),axis=0)
        y_test_five=y_test[y_test == label[5]]
        y_test_sorted=np.concatenate((y_test_sorted,5*np.ones((np.size(y_test_five),), dtype=int)),axis=0)

        y_train = keras.utils.to_categorical(y_train_sorted, num_classes)
        y_test = keras.utils.to_categorical(y_test_sorted, num_classes)

        model = cifar10vgg()

        predicted_x = model.predict(x_test_sorted)
        residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

        loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)
