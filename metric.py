# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:05:25 2020

@author: Guillaume
"""
import numpy as np
from keras import backend as K

def corr2(Img1,Img2): # correlation between the two images, metrics
    m_Img1 = np.mean(Img1)
    m_Img2 = np.mean(Img2)
    t1 = Img1-m_Img1
    t2 = Img2-m_Img2
    
    num = np.sum(np.multiply(t1,t2))
    den = np.sqrt(np.sum(t1**2)*np.sum(t2**2))
    r = num/den
    return r

def met_corr(batch_size):
    def corr(y_true, y_pred):
        r = 0
        for i in range(batch_size):
            m_img1 = K.mean(y_true[i, :, :, 0]) + 1e-6
            m_img2 = K.mean(y_pred[i, :, :, 0]) + 1e-6
            t1 = y_true[i, :, :, 0] - m_img1
            t2 = y_pred[i, :, :, 0] - m_img2

            num = K.sum(t1 * t2)
            den = K.sqrt(K.sum(t1 ** 2) * K.sum(K.square(t2)))
            r = r + num / den
        r /= batch_size
        return r
    return corr