"""
Created on Thu Apr 30 10:48:45 2020

@author: Guillaume
"""
import os
import matplotlib.pyplot as plt
import numpy as np
os.chdir('C:/Users/Guillaume/DL/project/Visibility_Correction')
from model import Unet, run_model
from bayesian_optimisation import run_bay_opti

path_data_train = 'C:/Users/Guillaume/DL/Data/training_feuille_exp_4/Tr4/'
path_data_test = 'C:/Users/Guillaume/DL/Data/training_feuille_exp_4/Ts/'
#path_data_train = '~/Tr/'
#path_data_test = '~/Ts/'
name_model = 'test'
path_data_model = 'C:/'
path_data_model = path_data_model + name_model +'/'
if not os.path.exists(path_data_model):
    os.mkdir(path_data_model)
if not os.path.exists(path_data_model):
    os.mkdir(path_data_model)
args = [name_model, path_data_train, path_data_test, path_data_model]
kwargs = {'epochs'             : 2,
          'pretrained_weights' : None, # weights location of the pretrained model
          'do'                 : .5,
          'weights_decay'      : 0.01, # for batch normalization layer
          'lr'                 : 1e-03,
          'momentum'           : 0.99, # for batch normalization layer
          'alpha_lr'           : 3e-02, # slope if leakyRelu
          'batch_size'         : 8,
          'depth'              : 5,
          'nb_unit'            : 64,
          'skip'               : 0,  #skip connection between input and output
          'nb_drop'            : 5,  # nb layer with dropout, must be odd and placed around the bottleneck
          'activation'         : 0,  #0 full relu 1 leakyrelu
          'unc'                : False,  # activate if prediction with uncertainty
          'end_act_func'       : 0,  # activation function of the last layer
          'loss_func'          : 0,
          'bay_opti'           : 0,  # if bayesian optimisation, the test set become the validation set
          'type_data'          : 0,  # 0 real 1 abs 2 real no rehauss
          'type_model'         : 1}  # 0 unet 1 resunet

_model = Unet(*args, **kwargs)
#%%
# train model
score = 1 - run_model(*args, **kwargs)

# simple prediction
_model = Unet(*args, **kwargs)
img1, img2, img3 = _model.model_predict(path_data_test, 0)
plt.subplot(311)
plt.imshow(img1[0, :, :, 0], cmap='hot')
plt.subplot(312)
plt.imshow(img2[0, :, :], cmap='hot')
plt.subplot(313)
plt.imshow(img3[0, :, :], cmap='hot')

# uncertainty prediction
kwargs['bay_opti'] = 0
kwargs['unc'] = True
_model = Unet(*args, **kwargs)
num_deep_ensembles = 10
num_examples = 17   # size test set
result = np.zeros((num_examples, 256, 256, 1))
patch_dim = 256

prediction_ensembles = np.ndarray((num_examples, patch_dim, patch_dim, 1, num_deep_ensembles))
for dropout_idx in range(num_deep_ensembles):
    for i in range(num_examples):
        result[i, :, :, :] = _model.model_predict(path_data_test, i, 0)[2]
    prediction_ensembles[:, :, :, :, dropout_idx] = result
    print('dropout ensembles: ' + str(dropout_idx + 1) + '/' + str(num_deep_ensembles))

f_result = np.ndarray((patch_dim, patch_dim, 2, num_examples))
for i in range(num_examples):
    f_result[:, :, 0, i] = np.mean(prediction_ensembles[i, :, :, 0, :].squeeze(), axis=2)
    f_result[:, :, 1, i] = np.std(prediction_ensembles[i, :, :, 0, :].squeeze(), axis=2)

ind = 11
x, y = _model.test_generator.__getitem__(ind)
plt.figure(1)
handle = plt.subplot(1, 5, 1)
handle.set_title('predicted mean')
plt.imshow(f_result[:, :, 0, ind], vmin=0, vmax=1, cmap='hot')
plt.axis('off')
handle = plt.subplot(1, 5, 2)
handle.set_title('predicted uncertainty')
plt.imshow(f_result[:, :, 1, ind], cmap='jet',vmin=np.mean(f_result[:, :, 1, ind]))
plt.axis('off')
handle = plt.subplot(1, 5, 3)
handle.set_title('ground truth')
plt.imshow(y.squeeze(), vmin=0, vmax=1, cmap='hot')
plt.axis('off')
handle = plt.subplot(1, 5, 4)
handle.set_title('absolute error')
plt.imshow(np.abs(y.squeeze() - f_result[:, :, 0, ind]), cmap='jet')
plt.axis('off')

# train model with bayesian optimisation

bounds = [
          {'name': 'do',            'type': 'continuous',    'domain': (0.0, 0.8)},
          {'name': 'weights_decay', 'type': 'continuous',    'domain': (0.0, 0.8)},
          {'name': 'lr',            'type': 'continuous',    'domain': (1e-5, 1e-3)},
          {'name': 'momentum',      'type': 'continuous',    'domain': (0.4, 1)},
          {'name': 'alpha_LR',          'type': 'continuous',    'domain': (0, .3)},
          {'name': 'batch_size',    'type': 'discrete',    'domain': (4, 8)},
          {'name': 'depth',         'type': 'discrete',    'domain': (3, 4, 5, 6, 7)},
          {'name': 'nb_unit',       'type': 'discrete',    'domain': (8, 16, 32, 64)},
          {'name': 'skip',          'type': 'discrete',    'domain': (0, 1)},
          {'name': 'nb_drop',       'type': 'discrete',    'domain': (0, 1, 2, 3, 4, 5)},
          {'name': 'end_act_func',       'type': 'discrete',    'domain': (0, 1, 2)},
          {'name': 'loss_func',       'type': 'discrete',    'domain': (0, 1)}
          ]
res = run_bay_opti(bounds, 40, 100,
               1,
               path_data_train, path_data_test, path_data_model, name_model, epochs = 2)

