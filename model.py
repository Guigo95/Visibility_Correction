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
                 name_model, path_data_train, path_data_test, path_data_model,
                 epochs=100, DO=.5, weights_decay=.01, lr=1e-4,
                 momentum=.99, alpha_LR=.1, batch_size=8, depth=5,
                 nb_unit=64, skip=0, nb_drop=5, activation=1, unc = False, bay_opti = 0,
                 end_act_func=0, loss_func=0, type_data=0, pretrained_weights=None):
        
        self.name_model = name_model
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
        self.unc = unc
        self.alpha_LR = alpha_LR
        self.type_data = type_data
        self.pretrained_weights = pretrained_weights
        if end_act_func == 0:
            self.end_act_func = None
        elif end_act_func == 1:
            self.end_act_func = 'relu'
        elif end_act_func == 2:
            self.end_act_func = 'sigmoid'
            
        if loss_func == 0 :
            self.loss_func = 'mse'
        else:
            self.loss_func = 'binary_crossentropy'
        self.bay_opti = bay_opti

        self.test_generator = self.prepare_test_data()
        self.__model = self.create_model()
        

    def prepare_training_data(self):
        params = {
                  'path_data' : self.path_data_train,
                  'batch_size': self.batch_size,
                  'type_data': self.type_data}

        name_file = os.listdir(self.path_data_train)
        random.seed(1)
        random.shuffle(name_file)
        split_v = (np.round(0.8 * len(name_file))).astype('int')
        data = {"train": name_file[0:split_v], "validation": name_file[split_v + 1:len(name_file) - 1]}
        training_generator = DataGenerator(data['train'], **params)
        validation_generator = DataGenerator(data['validation'], **params)           
        return training_generator, validation_generator

    def prepare_test_data(self):
        params = {
                  'shuffle' : False,
                  'path_data' : self.path_data_test,
                  'batch_size': 1,    
                  'type_data': self.type_data}
        
        name_file = os.listdir(self.path_data_test)
        data = {"test": name_file}
        test_generator = DataGenerator(data['test'], **params)
        return test_generator
        

    def create_block_down(self, model_tot, drop_layer, i_block):
        block = []
        unit = self.nb_unit * (2 ** i_block)

        if self.activation == 0:
            met_acti = 'relu'
            conva = Conv2D(unit, 3, activation=met_acti, padding='same', kernel_initializer='he_normal')(
                model_tot[-1][-1])
        else:
            conva = Conv2D(unit, 3, padding='same', kernel_initializer='he_normal')(model_tot[-1][-1])
            conva = LeakyReLU(alpha=self.alpha_LR)(conva)

        batchnorm = BatchNormalization(
            momentum=self.momentum,
            gamma_regularizer=l2(self.weights_decay),
            beta_regularizer=l2(self.weights_decay))(conva)
        if self.activation == 0:
            convb = Conv2D(unit, 3, activation=met_acti, padding='same', kernel_initializer='he_normal')(batchnorm)
        else:
            convb = Conv2D(unit, 3, padding='same', kernel_initializer='he_normal')(batchnorm)
            convb = LeakyReLU(alpha=self.alpha_LR)(convb)
        if drop_layer == 1:
            drop = Dropout(self.DO)(convb, training = self.unc)
            pool = MaxPooling2D(pool_size=(2, 2))(drop)
        else:
            pool = MaxPooling2D(pool_size=(2, 2))(convb)

        block.append(conva)
        block.append(batchnorm)
        block.append(convb)
        if drop_layer == 1:
            block.append(drop)
        block.append(pool)
        return block

    def create_block_up(self, model_tot, drop_layer, i_block):
        block = []
        unit = self.nb_unit * (2 ** (i_block - 1))

        if self.activation == 0:
            met_acti = 'relu'
            up = Conv2D(unit, 2, activation=met_acti, padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(model_tot[-1][-1]))
        else:
            up = Conv2D(unit, 2, padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(model_tot[-1][-1]))
            up = LeakyReLU(alpha=self.alpha_LR)(up)

        merge = concatenate([model_tot[i_block][-2], up], axis=3)

        if self.activation == 0:
            conva = Conv2D(unit, 3, activation=met_acti, padding='same', kernel_initializer='he_normal')(merge)
        else:
            conva = Conv2D(unit, 3, padding='same', kernel_initializer='he_normal')(merge)
            conva = LeakyReLU(alpha=self.alpha_LR)(conva)

        batchnorm = BatchNormalization(
            momentum=self.momentum,
            gamma_regularizer=l2(self.weights_decay),
            beta_regularizer=l2(self.weights_decay))(conva)

        if self.activation == 0:
            convb = Conv2D(unit, 3, activation=met_acti, padding='same', kernel_initializer='he_normal')(batchnorm)
        else:
            convb = Conv2D(unit, 3, padding='same', kernel_initializer='he_normal')(batchnorm)
            convb = LeakyReLU(alpha=self.alpha_LR)(convb)

        block.append(up)
        block.append(merge)
        block.append(conva)
        block.append(batchnorm)
        block.append(convb)

        if drop_layer == 1:
            drop = Dropout(self.DO)(convb, training = self.unc)
            block.append(drop)
        return block

    def create_model(self):
        model_tot = []
        inputs = Input((256, 256, 1))
        model_tot.append([inputs])
        # encoder
        for i in range(self.depth):
            if i > self.depth -(self.nb_drop-1)/2 -2:
                drop_layer = 1
            else:
                drop_layer = 0
            block_down = self.create_block_down(model_tot, drop_layer, i)
            model_tot.append(block_down)

        del model_tot[-1][-1]  # supress the last downsampling
        for i in range(self.depth - 1):
            if i < (self.nb_drop-1)/2 :
                drop_layer = 1
            else:
                drop_layer = 0
            block_up = self.create_block_up(model_tot, drop_layer, self.depth - (i + 1))
            model_tot.append(block_up)

        convend = Conv2D(1, 1, activation=self.end_act_func)(model_tot[-1][-1])

        model_tot.append([convend])
        if self.skip == 1:
            outputs = add([inputs, model_tot[-1][-1]])
        else:
            outputs = model_tot[-1][-1]
        model = Model(input=inputs, output=outputs)
     
        model_json = model.to_json()
        with open(self.path_data_model + self.name_model +'.json', 'w') as json_file:
            json_file.write(model_json)

        return model

    def model_fit(self):
        self.training_generator, self.validation_generator = self.prepare_training_data()
        
        check = ModelCheckpoint(self.path_data_model + self.name_model +'_corr' +'.h5', monitor='val_corr', save_best_only=True,
                                save_weights_only=True, mode='max')
        es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=0, mode='auto', baseline=None,
                           restore_best_weights=True)

        if (self.pretrained_weights):
            self.model.load_weights(self.pretrained_weights)

        self.__model.compile(optimizer=Adam(lr=self.lr), loss=self.loss_func, metrics=[met_corr(self.batch_size)])
        self.__model.fit_generator(generator=self.training_generator,
                                   validation_data=self.validation_generator,
                                   epochs=self.epochs, verbose=1, callbacks=[es, check])        
        self.__model = None

    def model_evaluate(self):
        self.model_fit()
        self.__model = model_from_json(open(self.path_data_model + self.name_model +'.json').read())
        self.__model.load_weights(self.path_data_model + self.name_model +'_corr' +'.h5')
        result_corr = []
       
        if self.bay_opti:
            self.test_generator = self.validation_generator # test the model on the validation set during the bayesiant process, and not on the test set

        for x , y in self.test_generator:            
            pred = self.__model.predict(x, verbose=1, steps=None, callbacks=None)
            result_corr.append (corr2(y, pred))

        evaluation = np.mean(np.array(result_corr))
        return 1 - evaluation

    def model_predict_from_testset(self, instance=0, load=1):
        if load == 1:
            self.__model = model_from_json(open(self.path_data_model + self.name_model +'.json').read())
        
        self.__model.load_weights(self.path_data_model + self.name_model +'_corr' +'.h5')
        
        x , y = self.test_generator.__getitem__(instance)
        pred = self.__model.predict(x, verbose=1, steps=None, callbacks=None)
        return x, y, pred

    def model_predict(self, path_data, instance, load=1):
        self.path_data_test = path_data
        self.test_generator = self.prepare_test_data()        
        return self.model_predict_from_testset(instance, load)


def run_model(*args,**kwargs):
    
    _model = Unet(*args, **kwargs)
    model_eval = _model.model_evaluate()
    del _model

    for i in range(3):
        gc.collect()
    K.clear_session()
    reset_keras()
    
    return model_eval

