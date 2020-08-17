# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:28:46 2020

@author: Guillaume
"""
import os
import gc
from data_generation import DataGenerator
from metric import corr2, met_corr
import random as random
import numpy as np
from keras.models import Model, model_from_json
from keras.layers import Input, MaxPooling2D, Conv2D, Dropout, UpSampling2D, concatenate, add, LeakyReLU
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.io.matlab import mio
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow


def reset_keras():
    sess = K.get_session()
    K.clear_session()
    sess.close()
    sess = K.get_session()

    print(gc.collect())

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    K.set_session(tensorflow.Session(config=config))





class Unet():
    def __init__(self,
                 path_data_train='', path_data_test='', path_data_model='',
                 epochs=150, DO=.5, weights_decay=.01, lr=1e-4,
                 momentum=.99, alpha_leaky_Relu=.1, batch_size=8, depth=5,
                 nb_unit=64, skip=0, nb_drop=2, activation=1,
                 end_act_func=0, loss_func=0, type_data=0, pretrained_weights=None):
        self.path_data_train = path_data_train
        self.path_data_test = path_data_test
        self.path_data_model = path_data_model
        self.nb_unit = nb_unit
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.DO = DO
        self.weights_decay = weights_decay
        self.batch_size = batch_size
        self.skip = skip
        self.nb_drop = nb_drop
        self.activation = activation
        self.alpha_leaky_Relu = alpha_leaky_Relu
        self.type_data = type_data
        self.pretrained_weights = pretrained_weights

        if end_act_func == 0:
            self.end_act_func = None
        elif end_act_func == 1:
            self.end_act_func = 'relu'
        elif end_act_func == 2:
            self.end_act_func = 'sigmoid'

        if loss_func == 0:
            self.loss_func = 'mse'
        else:
            self.loss_func = 'binary_crossentropy'

        self._X_test, self._y_test = self.test_data()
        self.__model = self.create_model()

    def prepare_data(self):
        params = {'dim': (256, 256),
                  'batch_size': self.batch_size,
                  'n_channels': 1,
                  'shuffle': True,
                  'pathdata': self.path_data_train,
                  'type_data': self.type_data}

        name_file = os.listdir(self.path_data_train)
        random.seed(1)
        random.shuffle(name_file)

        name_file = name_file[0:20]
        split_v = (np.round(0.8 * len(name_file))).astype('int')
        data = {"train": name_file[0:split_v], "validation": name_file[split_v + 1:len(name_file) - 1],
                "test": name_file[len(name_file) - 1]}
        training_generator = DataGenerator(data['train'], **params)
        validation_generator = DataGenerator(data['validation'], **params)
        x, y = training_generator.__getitem__(0)
        plt.subplot(1, 2, 1)
        plt.imshow((x[0, :, :, 0]), cmap='hot')
        plt.subplot(1, 2, 2)
        plt.imshow((y[0, :, :, 0]), cmap='hot')
        return training_generator, validation_generator

    def test_data(self):
        name_file = os.listdir(self.path_data_test)
        X_test = np.zeros((len(name_file), 256, 256, 1))
        y_test = np.zeros((len(name_file), 256, 256, 1))
        for i, ID in enumerate(name_file):
            data = mio.loadmat(self.path_data_test + ID)
            x_temp = data["X"]
            y_temp = data["Yf"]
            if self.type_data == 0:
                x_temp = np.real(x_temp) - np.min(np.real(x_temp))
                y_temp = np.real(y_temp) - np.min(np.real(y_temp))
            elif self.type_data == 1:
                x_temp = np.abs(x_temp)  #
                y_temp = np.abs(y_temp)  #
            x_temp = (x_temp / np.max(x_temp)).astype('float32')
            X_test[i,] = np.expand_dims(x_temp, axis=2)
            y_temp = (y_temp / np.max(y_temp)).astype('float32')
            y_test[i,] = np.expand_dims(y_temp, axis=2)

        return X_test, y_test

    def create_block_down(self, model_tot, drop_acti, i_block):
        block = []
        unit = self.nb_unit * (2 ** i_block)

        if self.activation == 0:
            met_acti = 'relu'
            conva = Conv2D(unit, 3, activation=met_acti, padding='same', kernel_initializer='he_normal')(
                model_tot[-1][-1])
        else:
            conva = Conv2D(unit, 3, padding='same', kernel_initializer='he_normal')(model_tot[-1][-1])
            conva = LeakyReLU(alpha=self.alpha_leaky_Relu)(conva)

        batchnorm = BatchNormalization(
            momentum=self.momentum,
            gamma_regularizer=l2(self.weights_decay),
            beta_regularizer=l2(self.weights_decay))(conva)
        if self.activation == 0:
            convb = Conv2D(unit, 3, activation=met_acti, padding='same', kernel_initializer='he_normal')(batchnorm)
        else:
            convb = Conv2D(unit, 3, padding='same', kernel_initializer='he_normal')(batchnorm)
            convb = LeakyReLU(alpha=self.alpha_leaky_Relu)(convb)
        if drop_acti == 1:
            drop = Dropout(self.DO)(convb)
            pool = MaxPooling2D(pool_size=(2, 2))(drop)
        else:
            pool = MaxPooling2D(pool_size=(2, 2))(convb)

        block.append(conva)
        block.append(batchnorm)
        block.append(convb)
        if drop_acti == 1:
            block.append(drop)
        block.append(pool)
        return block

    def create_block_up(self, model_tot, drop_acti, i_block):
        block = []
        unit = self.nb_unit * (2 ** (i_block - 1))

        if self.activation == 0:
            met_acti = 'relu'
            up = Conv2D(unit, 2, activation=met_acti, padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(model_tot[-1][-1]))
        else:
            up = Conv2D(unit, 2, padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(model_tot[-1][-1]))
            up = LeakyReLU(alpha=self.alpha_leaky_Relu)(up)

        merge = concatenate([model_tot[i_block][-2], up], axis=3)

        if self.activation == 0:
            conva = Conv2D(unit, 3, activation=met_acti, padding='same', kernel_initializer='he_normal')(merge)
        else:
            conva = Conv2D(unit, 3, padding='same', kernel_initializer='he_normal')(merge)
            conva = LeakyReLU(alpha=self.alpha_leaky_Relu)(conva)

        batchnorm = BatchNormalization(
            momentum=self.momentum,
            gamma_regularizer=l2(self.weights_decay),
            beta_regularizer=l2(self.weights_decay))(conva)

        if self.activation == 0:
            convb = Conv2D(unit, 3, activation=met_acti, padding='same', kernel_initializer='he_normal')(batchnorm)
        else:
            convb = Conv2D(unit, 3, padding='same', kernel_initializer='he_normal')(batchnorm)
            convb = LeakyReLU(alpha=self.alpha_leaky_Relu)(convb)

        block.append(up)
        block.append(merge)
        block.append(conva)
        block.append(batchnorm)
        block.append(convb)

        if drop_acti == 1:
            drop = Dropout(self.DO)(convb)
            block.append(drop)
        return block

    def create_model(self):
        model_tot = []
        inputs = Input((256, 256, 1))
        model_tot.append([inputs])
        # encoder
        for i in range(self.depth):
            if i > self.nb_drop:
                drop_acti = 1
            else:
                drop_acti = 0
            block_down = self.create_block_down(model_tot, drop_acti, i)
            model_tot.append(block_down)

        del model_tot[-1][-1]  # supress the last downsampling
        for i in range(self.depth - 1):
            if i > self.nb_drop:
                drop_acti = 0
            else:
                drop_acti = 1
            block_up = self.create_block_up(model_tot, drop_acti, self.depth - (i + 1))
            model_tot.append(block_up)

        conv10 = Conv2D(1, 1, activation=self.end_act_func)(model_tot[-1][-1])

        model_tot.append([conv10])
        if self.skip == 1:
            outputs = add([inputs, model_tot[-1][-1]])
        else:
            outputs = model_tot[-1][-1]
        model = Model(input=inputs, output=outputs)
        # model.summary()
        model_json = model.to_json()
        with open(self.path_data_model + 'best_weights_corr.json', 'w') as json_file:
            json_file.write(model_json)

        return model

    def model_fit(self):
        self.__training_generator, self.__validation_generator = self.prepare_data()
        check = ModelCheckpoint(self.path_data_model + 'best_weights_corr.h5', monitor='val_corr', save_best_only=True,
                                save_weights_only=True, mode='max')
        es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=0, mode='auto', baseline=None,
                           restore_best_weights=True)

        if (self.pretrained_weights):
            self.__model.load_weights(self.pretrained_weights + 'best_weights_corr.h5')

        self.__model.compile(optimizer=Adam(lr=self.lr), loss=self.loss_func, metrics=[met_corr(self.batch_size)])

        self.__model.fit_generator(generator=self.__training_generator,
                                   validation_data=self.__validation_generator,
                                   epochs=self.epochs, verbose=1, callbacks=[es, check])

        self.__model = None

    def model_evaluate(self):
        self.model_fit()
        model_p = model_from_json(open(self.path_data_model + 'best_weights_corr.json').read())
        model_p.load_weights(self.path_data_model + 'best_weights_corr.h5')
        n_testset = self._X_test.shape[0]
        result_corr = np.zeros((n_testset, 256, 256, 1))
        img2 = self._y_test

        for i in range(0, n_testset):
            img1 = model_p.predict(self._X_test[i, :, :, :], verbose=1, steps=None, callbacks=None)
            result_corr[i] = corr2(img1[i, :, :, 0], img2[i, :, :, 0])

        evaluation = np.mean(result_corr)
        del model_p
        return 1 - evaluation

    def model_predict_from_testset(self, instance=0, load=1):
        if load == 1:
            model_p = model_from_json(open(self.path_data_model + 'best_weights_corr.json').read())
            model_p.load_weights(self.path_data_model + 'best_weights_corr.h5')

        Img1 = model_p.predict(self._X_test[instance, :, :, :], verbose=1, steps=None, callbacks=None)
        Img2 = self._y_test[instance, :, :, :]
        Img3 = self._X_test[instance, :, :, :]
        return Img1, Img2, Img3

    def model_predict(self, path_data, instance, load=1):
        instance = list(instance)
        name_file_tot = os.listdir(path_data)
        name_file = [name_file_tot[i] for i in instance]
        X = np.zeros((len(name_file), 256, 256, 1))
        for i, ID in enumerate(name_file):
            data = mio.loadmat(path_data + ID)
            x_temp = data["X"]
            if self.type_data == 0:
                x_temp = np.real(x_temp) - np.min(np.real(x_temp))
            elif self.type_data == 1:
                x_temp = np.abs(x_temp)  #
            x_temp = x_temp / np.max(x_temp)
            x_temp = x_temp.astype('float32')
            X[i,] = np.expand_dims(x_temp, axis=2)

        if load == 1:
            model_p = model_from_json(open(self.path_data_model + 'best_weights_corr.json').read())
            model_p.load_weights(self.path_data_model + 'best_weights_corr.h5')

        Img1 = model_p.predict(X, verbose=1, steps=None, callbacks=None)
        Img2 = X
        return Img1, Img2


def run_model(path_data_model=None,
              path_data_train=None,
              path_data_test=None,
              epochs=150, DO=.5, weights_decay=.01, lr=1e-4,
              momentum=.99, taux=.1, batch_size=8, depth=5, nb_unit=64,
              skip=0, nb_drop=2, activation=1, last_acti=None,
              loss_func=0, type_data=0,
              pretrained_weights=None):
    _model = Unet(path_data_model=path_data_model,
                       path_data_train=path_data_train,
                       path_data_test=path_data_test,
                       epochs=epochs, DO=DO, weights_decay=weights_decay, lr=lr,
                       momentum=momentum,
                       alpha_leaky_Relu=taux, batch_size=batch_size, depth=depth,
                       nb_unit=nb_unit, skip=skip, nb_drop=nb_drop,
                       activation=activation, end_act_func=last_acti,
                       loss_func=loss_func,
                       type_data=type_data,
                       pretrained_weights=None)
    model_eval = _model.model_evaluate()
    del _model

    for i in range(3):
        gc.collect()
    K.clear_session()
    reset_keras()
    return model_eval

