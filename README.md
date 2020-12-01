# Visibility_Correction

UNET implementation, similar than the one used for the paper: 'Compensating for visibility artefacts in photoacoustic imaging with a deep learning approach providing prediction uncertainties' 
In this paper a model has been trained to remove artefacts on 2D photoacoustic images, and produce an uncertainty map of the prediction.
![alt text](https://github.com/Guigo95/Visibility_Correction/img/github_result.JPG)
A bayesian optimisation approach has been implemented in order to find the best parameters of the model.

To implement the method, it is required to set the following parameters in the main file: path_data_train (location of the training set), path_data_test (location of the test set) and path_data_model  (the location where the model will be saved).

Note: input data must be in mat format.
