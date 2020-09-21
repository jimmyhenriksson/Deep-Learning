import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from Stacked_Model import *
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def retrieve_lookback(X,Y,lookback):
    n_samples, dimensions = X.shape
    X = np.asarray([X[i-5:i,:].flatten() for i in range(lookback,n_samples)])
    Y = np.asarray([Y[i] for i in range(lookback,n_samples)])

    return X, Y


def run_experiment(model,X, Y, epochs=100):
    model.compile()
    model.train(X,Y,epochs=epochs)
    model.plot()

    
def load_data(lookback, lookahead):
    X_train = np.load(f'../data/processed/OMXS_processed_forecast{lookahead}_train.npy')
    y_train = np.load(f'../data/processed/OMXS_labels_forecast{lookahead}_train.npy')
    X_test = np.load(f'../data/processed/OMXS_processed_forecast{lookahead}_test.npy')
    y_test = np.load(f'../data/processed/OMXS_labels_forecast{lookahead}_test.npy')

    X, Y = retrieve_lookback(X_train,y_train,lookback)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.20)
    
    return X_train, y_train, X_test, y_test, X_val, y_val    

def gan_experiment(X_train, y_train, X_val, y_val, lookahead, gan_epochs = 50, gan_hidden_size =100, SM_epochs = 50, batch_size = 64):


    """
    GAN
    """
    noise_dim = X_train.shape[1]
    gan = GAN_FINANCE(gan_hidden_size,output_size=noise_dim, batch_size=batch_size, noise_dim=noise_dim)
    X_gan = copy.deepcopy(X_train)
    X_gan = tf.data.Dataset.from_tensor_slices(X_gan).batch(batch_size)
    gan.train(X_gan, epochs = gan_epochs, verbose=False)

    """
    Experimental stacked LSTM
    """
    number_of_samples, dim = X_train.shape
    experimental = Experimental(dim, dim, lookahead, gan)
    experimental.compile()
    experimental.train(X_train, y_train, X_val, y_val, epochs=SM_epochs, batch_size=batch_size)
    
    experimental.plot()
    
    return experimental

if __name__ == "__main__":
    X = np.load('../data/processed/OMXS_processed.npy')
    Y = np.load('../data/processed/OMXS_labels.npy')

    X, Y = retrieve_lookback(X,Y,5)