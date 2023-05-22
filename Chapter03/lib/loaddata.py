# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 09:35:02 2021

@author: RPL 2020
"""

from lib import bdtb
from scipy.io import savemat, loadmat
import numpy as np
from sklearn import preprocessing

def fromArch(i=0):
    # In[]:
    matlist=[]
    matlist.append('./de_s1_V1_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_leave0_1x1_preprocessed.mat')
    #matlist.append('../de_s1_V2_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_leave0_1x1_preprocessed.mat')
    #matlist.append('../de_s1_V1V2_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_leave0_1x1_preprocessed.mat')
    #matlist.append('../de_s1_V3VP_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_leave0_1x1_preprocessed.mat')
    #matlist.append('../de_s1_AllArea_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_leave0_1x1_preprocessed.mat')
    matfile=matlist[0]
    # In[]: membangun model berdasarkan MLP arsitektur berikut(dalam list) 
    archl=[]
    archl.append('784_256_128_10')
    archl.append('784_256_128_6')
    archl.append('784_256_128_5')
    archl.append('200_100')
    archl.append('1')
    matfile=matlist[0]#memilih satu file saja V1
    
    # In[]: train and predict rolly
    arch=archl[i]
    label,pred=bdtb.testModel(matfile,arch)
    allscoreresults=bdtb.simpanScore(label, pred, matfile, arch)
    return label,pred,allscoreresults

def Miyawaki():
    predm,labelm,scorem=bdtb.simpanScoreMiyawaki()
    return labelm,predm,scorem

def Data28():
    # In[]: Load dataset
    handwriten_69=loadmat('digit69_28x28.mat')
    #ini fmri 10 test 90 train satu baris berisi 3092
    Y_train = handwriten_69['fmriTrn'].astype('float32')
    Y_test = handwriten_69['fmriTest'].astype('float32')

    # ini stimulus semua
    X_train = handwriten_69['stimTrn']#90 gambar dalam baris isi per baris 784 kolom
    X_test = handwriten_69['stimTest']#10 gambar dalam baris isi 784 kolom
    # normalisasi agar hasil 0-1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # In[]: X adalah gambar stimulus,ukuran pixel 28x28 = 784 di flatten sebelumnya dalam satu baris, 28 row x 28 column dengan channel 1(samaa kaya miyawaki)
    resolution = 28
    X_train = X_train.reshape([X_train.shape[0], resolution, resolution])
    X_test = X_test.reshape([X_test.shape[0], resolution, resolution])
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))
    #X_train = X_train.reshape([X_train.shape[0], resolution, resolution])
    #X_test = X_test.reshape([X_test.shape[0], resolution, resolution])
    #channel di depan
    #X_train = X_train.reshape([X_train.shape[0], 1, resolution, resolution])
    #X_test = X_test.reshape([X_test.shape[0], 1, resolution, resolution])
    #channel di belakang(edit rolly) 1 artinya grayscale
    X_train = X_train.reshape([X_train.shape[0], resolution, resolution, 1])
    X_test = X_test.reshape([X_test.shape[0], resolution, resolution, 1])
    # In[]: Normlization sinyal fMRI, min max agar nilainya hanya antara 0 sd 1
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))   
    Y_train = min_max_scaler.fit_transform(Y_train)     
    Y_test = min_max_scaler.transform(Y_test)

    print ('X_train.shape : ')
    print (X_train.shape)
    print ('Y_train.shape')
    print (Y_train.shape)
    print ('X_test.shape')
    print (X_test.shape)
    print ('Y_test.shape')
    print (Y_test.shape)
    numTrn=X_train.shape[0]#ada 90 data training
    numTest=X_test.shape[0]#ada 10 data testing
    return X_train,X_test,Y_train,Y_test