import numpy as np
import os
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import tensorflow.python.keras as K
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Conv1D, Conv2D, Activation, MaxPooling1D, Flatten, Dense,\
                         LSTM, BatchNormalization, LeakyReLU, Conv2DTranspose, Reshape, Dropout, Input, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from abc import ABC
import sys
import copy



class Model(ABC):
    def __init__(self):
        pass
        

    def build(self):
        self.model = None


    def compile(self):
        self.build()
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        self.model.build()


    def train(  self, 
                X, \
                Y, \
                validation_split = 0.10, \
                shuffle = True, \
                batch_size = 100, \
                epochs=10, \
                early_stopping = False,\
            ):

        if early_stopping:
            es = EarlyStopping(monitor='val_binary_accuracy', mode='auto', patience=50)
            self.history = self.model.fit(   X, \
                                        Y,\
                                        validation_split=validation_split,\
                                        shuffle=shuffle,\
                                        batch_size=batch_size,\
                                        epochs=epochs,\
                                        callbacks = [es],\
                                        verbose = 0,\

                                    )
        else:
            self.history = self.model.fit(   X, \
                                        Y,\
                                        validation_split=validation_split,\
                                        shuffle=shuffle,\
                                        batch_size=batch_size,\
                                        epochs=epochs,\
                                        verbose = 0,\
                                        )
        return self.history


    def get_accuracy(self, X, Y):
        predictions = self.model.predict(X)
        preds = np.where(predictions>0.50,1,0)
        n_samples = preds.shape[0]
        accuracy = np.sum(preds == Y)/n_samples
        return accuracy


    def plot(self):
        validation_loss = self.history.history['val_loss']
        training_loss = self.history.history['loss']
        training_accuracy = self.history.history['binary_accuracy']
        validation_accuracy = self.history.history['val_binary_accuracy']
        plt.plot(training_loss, label="Training Loss")
        plt.plot(validation_loss, label="Validation Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

        plt.plot(training_accuracy, label="Training Accuracy")
        plt.plot(validation_accuracy, label="Validation Accuracy")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()
        

class MLP(Model):
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden
        

    def build(self):
        """
        Designs an MLP with one hidden layer with n_hidden nodes
        """
        model = Sequential()
        model.add(Dense(self.n_hidden, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        self.model = model
        return model


class Deep_MLP(Model):
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden
        

    def build(self):
        """
        Designs an MLP with one hidden layer with n_hidden nodes
        """
        model = Sequential()
        model.add(Dense(self.n_hidden, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.30))
        model.add(Dense(self.n_hidden,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.30))
        model.add(Dense(self.n_hidden,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.30))
        model.add(Dense(1, activation='sigmoid'))
        self.model = model
        return model


class CNN(Model):
    def __init__(self, kernel_size=2, pool_size =2):
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        pass


    def build(self):
        stride=self.kernel_size

        model = Sequential()
        model.add(Conv1D(64, kernel_size=self.kernel_size, strides=stride, activation='relu'))
        model.add(MaxPooling1D(pool_size=self.pool_size,strides=stride))
        model.add(Dropout(0.3))
        model.add(Conv1D(32, kernel_size=self.kernel_size, strides=stride, activation='relu'))
        model.add(MaxPooling1D(pool_size=self.pool_size,strides=stride))
        model.add(Flatten())
        model.add(Dense(100,activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        self.model = model
        return model

    def train(  self, 
                X, \
                Y,\
                epochs=100
            ):
        n_samples, dimensions = X.shape
        X = X.reshape((n_samples,dimensions,1))
        super().train(X,Y,epochs=epochs)


    def get_accuracy(self, X, Y):
        n_samples, dimensions = X.shape
        X = X.reshape((n_samples,dimensions,1))
        accuracy = super().get_accuracy(X,Y)
        return accuracy



class CNN_LSTM(Model):
    def __init__(self, kernel_size=2, pool_size =2):
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        pass


    def build(self):
        stride=self.kernel_size

        model = Sequential()
        model.add(Conv1D(64, kernel_size=self.kernel_size, strides=stride, activation='relu'))
        model.add(MaxPooling1D(pool_size=self.pool_size,strides=stride))
        model.add(Dropout(0.3))
        model.add(Conv1D(32, kernel_size=self.kernel_size, strides=stride, activation='relu'))
        model.add(MaxPooling1D(pool_size=self.pool_size,strides=stride))
        model.add(LSTM(100))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))
        self.model = model
        return model


    def train(  self, 
                X, \
                Y,\
                epochs=100
            ):
        n_samples, dimensions = X.shape
        X = X.reshape((n_samples,dimensions,1))
        super().train(X,Y,epochs=epochs)


    def get_accuracy(self, X, Y):
        n_samples, dimensions = X.shape
        X = X.reshape((n_samples,dimensions,1))
        accuracy = super().get_accuracy(X,Y)
        return accuracy


class LSTM_(Model):
    def __init__(self, units):
        self.units = units        


    def build(self):
        model = Sequential()
        model.add(LSTM(self.units, input_shape=(None,1)))
        model.add(Dense(1, activation="sigmoid"))
        self.model = model
        return model

    def compile(self):
        self.build()
        opt = Adam(learning_rate=0.0001)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
        self.model.build()

    
    def train(  self, 
                X, \
                Y,\
                epochs=100
            ):
        n_samples, dimensions = X.shape
        X = X.reshape((n_samples,dimensions,1))
        super().train(X,Y,epochs=epochs)


    def get_accuracy(self, X, Y):
        n_samples, dimensions = X.shape
        X = X.reshape((n_samples,dimensions,1))
        accuracy = super().get_accuracy(X,Y)
        return accuracy


class CNN_LSTM_Regularized(Model):
    def __init__(self, kernel_size=2, pool_size =2):
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        pass

    def compile(self):
        self.build()
        opt = Adam(learning_rate=0.0001)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
        self.model.build()


    def build(self):
        stride=self.kernel_size

        model = Sequential()
        model.add(Conv1D(64, kernel_size=self.kernel_size, strides=stride, activation='relu'))
        model.add(MaxPooling1D(pool_size=self.pool_size,strides=stride))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Conv1D(32, kernel_size=self.kernel_size, strides=stride, activation='relu'))
        model.add(MaxPooling1D(pool_size=self.pool_size,strides=stride))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(100))
        model.add(Dropout(0.3))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        self.model = model
        return model

if __name__ == "__main__":
    # X = np.ones((100,100))
    # Y = np.ones((100,1))

    X = np.asarray([[0,0],
                    [0,1],
                    [1,0],
                    [1,1],
                    ])
    Y = np.asarray([[0],[1],[1],[1],])

    """
    MLP
    """
    # mlp = MLP(100)
    # mlp.compile()
    # mlp.train(X,Y,epochs=200)
    # print("Test accuracy: ", mlp.get_accuracy(X,Y))
    # mlp.plot()


    """
    Deep MLP
    """
    d_mlp = Deep_MLP(100)
    d_mlp.compile()
    d_mlp.train(X,Y,epochs=100)
    print("Test accuracy: ", d_mlp.get_accuracy(X,Y))
    d_mlp.plot()


    """
    CNN
    """
    # CNN expects the data to be in a particular shape
    # 4 dimensions, (number of samples, dimensions, time, 1)
    # X = np.ones((100,100,100))
    # n,d,t = X.shape
    # X = X.reshape((n,d,t,1))
    # Y = np.ones((100,1))
    # n,d = X.shape
    # X = X.reshape((n,d,1))
    # cnn = CNN()
    # cnn.compile()
    # cnn.train(X,Y)
    # cnn.plot()
    # print(cnn.get_accuracy(X,Y))

    """
    LSTM
    # """
    # lstm = LSTM_(100)
    # lstm.compile()
    # lstm.train(X,Y)
    # lstm.plot()
    # print(lstm.get_accuracy(X,Y))