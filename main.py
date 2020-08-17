# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:48:45 2020

@author: Guillaume
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from model import Unet, run_model
from bayesian_optimisation import bay_opti
from metric import corr2

path_data_train = 'C:/Users/Guillaume/DL/Data/training_feuille_exp_4/Tr/'
path_data_test = 'C:/Users/Guillaume/DL/Data/training_feuille_exp_4/Ts/'
name_model = 'pretained_unet'
path_data_model =  'D:/Guillaume/reduce_dataset/simu2exp/'
path_data_model = path_data_model + name_model
path_data_project = 'C:/Users/Guillaume/DL/project/reduce_dataset/simu2exp/scripts'

os.chdir(path_data_project)
if not os.path.exists(path_data_model):
    os.mkdir(path_data_model)

inputs = {'epochs'          : 150,
          'path_data_train' : path_data_train,
          'path_data_test' : path_data_test,
          'path_data_model' : path_data_model,
          'DO'              : 3.55372162e-01,
          'weights_decay'   : 5.26850274e-01,
          'lr'              : 1.00000000e-03,
          'momentum'        : 9.17815077e-01,
          'taux'            : 2.91754857e-02,
          'batch_size'      : 8,
          'depth'           : 5, 
          'nb_unit'         : 64, 
          'skip'            : 0, 
          'nb_drop'         : 3, 
          'activation'      : 1, #0 full relu 1 leakyrelu (=relu if taux = 0) 
          'last_acti'       : 2, # 0 None 1 relu 2 sigmoid 
          'loss_func'       : 1, # 0 mse 1 binary_crossentropy
          'type_data'       : 1} # 0 real 1 abs 2 real no rehauss


# train model
score = 1 - run_model(**inputs)

# prediction
n_testset = 17
_model = Unet(**inputs)
img1, img2 = _model.model_predict(range(n_testset))
plt.subplot(211)
plt.imshow(img1[0, :, :, 0], cmap='hot')
plt.subplot(212)
plt.imshow(img2[0, :, :], cmap='hot')
# %%
result_corr = np.zeros(n_testset)

count = 0
for i in range(n_testset):

    result_corr[count] = corr2(img1[i, :, :, 0], img2[i, :, :])
    count = count + 1

score = np.mean(result_corr)

#%% train model with bayesian optimisation
bounds = [
          {'name': 'DO',            'type': 'continuous',    'domain': (0.0, 0.8)},
          {'name': 'weights_decay', 'type': 'continuous',    'domain': (0.0, 0.8)},
          {'name': 'lr',            'type': 'continuous',    'domain': (1e-3, 1e-5)},
          {'name': 'momentum',      'type': 'continuous',    'domain': (0.4, 1)},
          {'name': 'taux',          'type': 'continuous',    'domain': (0, .3)},
          {'name': 'batch_size',    'type': 'discrete',    'domain': (4, 8)},
          {'name': 'depth',         'type': 'discrete',    'domain': (3, 4, 5, 6, 7)},
          {'name': 'nb_unit',       'type': 'discrete',    'domain': (8, 16, 32, 64)},
          {'name': 'skip',          'type': 'discrete',    'domain': (0, 1)},
          {'name': 'nb_drop',       'type': 'discrete',    'domain': (0, 1, 2, 3, 4, 5)},
          {'name': 'last_acti',       'type': 'discrete',    'domain': (0, 1, 2)},
          {'name': 'loss_func',       'type': 'discrete',    'domain': (0, 1)}
          #{'name': 'activation',    'type': 'discrete',    'domain': (0, 1)}
          ] 
res = bay_opti(bounds, 40, 40*3600, 
               1, 
               path_data_train, path_data_test, path_data_model)
