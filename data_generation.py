# -*- coding: utf-8 -*-
"""
@author: GODEFROY Guillaume
"""

import numpy as np
import keras
from scipy.io.matlab import mio


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_ids, path_data,  batch_size = 8, dim = (256, 256), n_channels = 1,
                 shuffle = True,  type_data = 0):
        # Initializaion
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.path_data = path_data
        self.type_data = type_data
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        # Generates data containing batch_size samples, X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))
        # y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_ids_temp):

            data = mio.loadmat(self.path_data + ID)
            x_temp = data["X"]
            y_temp = data["Yf"]

            if self.type_data == 0: # real, positif
                x_temp = np.real(x_temp) - np.min(np.real(x_temp))
                y_temp = np.real(y_temp) - np.min(np.real(y_temp))
            elif self.type_data == 1: # module
                x_temp = np.abs(x_temp)
                y_temp = np.abs(y_temp)

            if self.type_data == 2: # real
                x_temp = np.real(x_temp)
                y_temp = np.real(y_temp)

            x_temp = x_temp / np.max(x_temp)
            x_temp = x_temp.astype('float32')
            x[i, ] = np.expand_dims(x_temp, axis=2)

            y_temp = y_temp / np.max(y_temp)
            y_temp = y_temp.astype('float32')
            y[i, ] = np.expand_dims(y_temp, axis=2)

        return x, y
