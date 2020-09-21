import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow.python.keras as keras
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras import layers
# from tensorflow.python.keras.layers import Conv1D, Conv2D, Activation, MaxPooling1D, Flatten, Dense,\
#                          LSTM, BatchNormalization, LeakyReLU, Conv2DTranspose, Reshape, Dropout, Input, concatenate
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# from tensorflow.python.keras.optimizers import Adam


import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, Conv2D, Activation, MaxPooling1D, Flatten, Dense,\
                         LSTM, BatchNormalization, LeakyReLU, Conv2DTranspose, Reshape, Dropout, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from abc import ABC
import sys
import copy

from Models import Model
        

class GAN(Model):
    def __init__(self, n_hidden, output_size, batch_size=100, noise_dim = 100):
        self.BATCH_SIZE = batch_size
        self.noise_dim = noise_dim

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)

    
    def build_generator(self):
        """
        TODO: Consider different topology
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    
    def build_discriminator(self):
        """
        TODO: Consider different topology
        """
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    
    def compile(self):
        pass

    @tf.function
    def train_step(self, images):

        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim ])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise)

            real_output = self.discriminator(images)
            fake_output = self.discriminator(generated_images)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    
    def train(self, dataset, epochs=50):
        for epoch in range(epochs):
            print(epochs)

            for image_batch in dataset:
                self.train_step(image_batch)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)


class GAN_FINANCE(Model):
    def __init__(self, n_hidden, output_size, batch_size=100, noise_dim = 100):
        self.n_hidden = n_hidden
        self.BATCH_SIZE = batch_size
        self.noise_dim = noise_dim

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)

    
    def build_generator(self):
        """
        TODO: Consider different topology
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(self.n_hidden))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(self.n_hidden))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(self.n_hidden))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Generate an output of the sample size
        model.add(layers.Dense(self.noise_dim ))

        return model

    
    def build_discriminator(self):
        """
        TODO: Consider different topology
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(self.n_hidden))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(self.n_hidden))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, sample, verbose = False):

        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim ])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_sample = self.generator(noise)

            if verbose:
                tf.print(generated_sample, output_stream=sys.stdout)

            real_output = self.discriminator(sample)
            fake_output = self.discriminator(generated_sample)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    
    def train(self, dataset, epochs=500, verbose = False):
        for epoch in range(epochs):
            if verbose: 
                print("epoch:", epoch)

            for sample_batch in dataset:
                self.train_step(sample_batch, verbose=verbose)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)


    def generate(self, lookahead=90, noise = None):
#         generated_samples = np.zeros((lookahead,self.noise_dim))

        if noise is None:
            noise = tf.random.normal([1, self.noise_dim ])

        noise = self.generator(noise)
        generated_samples = noise

        return generated_samples


class Experimental(Model):
    def __init__(self, dim_1, dim_2, lookahead, GAN, batch_size=25):
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.lookahead = lookahead
        self.GAN = GAN
        self.batch_size = batch_size
        pass


    def build(self):

        lstm1_input = Input(shape=(None,1))
        x = LSTM(32)(lstm1_input)
        lstm1_output = Dense(1)(x)

        lstm2_input = Input(shape=(self.lookahead,self.dim_2)) #TODO Check the shapes here
        x = LSTM(32)(lstm2_input)
        lstm2_output = Dense(1)(x)

        concat = concatenate([lstm1_output, lstm2_output], axis=1)
        x = Dense(100, activation="relu")(concat)
        x = Dense(100, activation="relu")(x)
        output = Dense(1,activation="sigmoid")(x)

        model = keras.models.Model(inputs=[lstm1_input, lstm2_input], outputs=[output])
        model.summary()
        
        self.model = model
        return model


    def train(  self, 
                actual_samples, \
                Y, \
                X_val,\
                Y_val,\
                validation_split = 0.10, \
                shuffle = True, \
                batch_size = 256, \
                epochs=2, \
                early_stopping = True,\
                ratio = 0.90,\
                lookahead = 5,\
            ):

        n_samples, dimensions = actual_samples.shape
        actual_samples = actual_samples.reshape((n_samples,dimensions,1))

        n_val_samples, val_dim = X_val.shape
        X_val_gan = np.asarray([self.GAN.generate(lookahead=lookahead, noise = x.reshape(1,-1)) for x in X_val])

        self.history = self.model.fit_generator( generator =self.train_generator(actual_samples, Y, batch_size, lookahead),\
                                            validation_data=([X_val, X_val_gan], Y_val),\
                                            steps_per_epoch=epochs/batch_size,\
                                            epochs=epochs
                                            )

        return self.history


    def train_generator(self, actual_samples, labels, lookahead, batch_size=256):
        """
        MFCCs have a variable number of frames so we pad each batch to the max number of 
        timesteps per batch to allow our LSTM to train
        args:
            X (list): Nx (39,_variable dimension) list
            Y (np.matrix): N   label matrix
        returns:
            X_batch (np.matrix): features of training batch
            Y_batch (np.matrix): labels of training batch
        """
        n_samples, n_dim, _ = actual_samples.shape
        indices = np.arange(n_samples)
        

        while True:
            # randomly take a batch of batch_size from training data
            samples = np.random.choice(indices, batch_size)
            
            # Generate a trajectory for each sample
            generated_samples = np.asarray([self.GAN.generate(lookahead=lookahead, noise = actual_samples[i].T) for i in samples])
            
            # Obtain the labels
            x_batch = actual_samples[samples]
            y_batch = labels[samples]


            yield [x_batch, generated_samples ], y_batch


    def compile(self):
        self.build()
        opt = Adam(learning_rate=0.0001)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])


    def get_accuracy(self, X, Y, lookahead):
        generated = np.asarray([self.GAN.generate(lookahead=lookahead, noise = i.reshape(1,-1)) for i in X])
        predictions = self.model.predict([X,generated])
        preds = np.where(predictions>0.50,1,0)
        n_samples = preds.shape[0]
        accuracy = np.sum(preds == Y)/n_samples
        return accuracy


if __name__ == "__main__":
    X = np.ones((100,100))
    n,d = X.shape
    X_val = np.ones((20,100))

    Y = np.ones((100,1))
    Y_val = np.ones((20,1))


    """
    GAN FINANCE
    # """
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    noise_dim = X.shape[1]
    gan = GAN_FINANCE(100,output_size=noise_dim, batch_size=BATCH_SIZE, noise_dim=noise_dim)
    X_gan = copy.deepcopy(X)
    X_gan = tf.data.Dataset.from_tensor_slices(X_gan).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    gan.train(X_gan, epochs = 5, verbose=False)


    """
    Experimental stacked LSTM
    """
    number_of_samples, s_LSTM_dim = X.shape

    g = np.random.randn(number_of_samples,90,s_LSTM_dim)
    number_of_samples, lookahead, g_LSTM_dim = g.shape

    experimental = Experimental(g_LSTM_dim, s_LSTM_dim, lookahead, gan)
    experimental.compile()

    experimental.train(X,Y, X_val, Y_val)
