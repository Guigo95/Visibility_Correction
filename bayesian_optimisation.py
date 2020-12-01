# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:06 2020

@author: Guillaume
"""
from model import run_model
import GPyOpt


def run_bay_opti(bounds, max_iter, max_time, type_data, path_data_train, path_data_test, path_data_model, name_model, epochs=100):
    def f(x):
        print(x)
        evaluation = run_model(
            do=float(x[:, 0]),
            weights_decay=float(x[:, 1]),
            lr=float(x[:, 2]),
            momentum=float(x[:, 3]),
            alpha_lr=float(x[:, 4]),
            batch_size=int(x[:, 5]),
            depth=int(x[:, 6]),
            nb_unit=int(x[:, 7]),
            skip=int(x[:, 8]),
            nb_drop=int(x[:, 9]),
            end_act_func=int(x[:, 10]),
            loss_func=int(x[:, 11]),
            type_data=type_data,
            path_data_train=path_data_train,
            path_data_test=path_data_test,
            path_data_model=path_data_model,
            name_model=name_model,
            bay_opti=1,
            unc=False,
            epochs=epochs
            #   activation = int(x[:,10]),
        )

        # print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation))
        print(evaluation)
        return evaluation

    opt_feuille = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
    opt_feuille.run_optimization(max_iter=max_iter, max_time=max_time, verbosity=True)
    opt_feuille.save_report(path_data_model + 'report.txt')
    opt_feuille.save_evaluations(path_data_model + 'eval.txt')
    opt_feuille.save_models(path_data_model + 'model.txt')

    return opt_feuille.x_opt
