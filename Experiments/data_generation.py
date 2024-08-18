import pandas as pd
import csv
from pandas import read_csv
import copy
import numpy as np
import os

#"1, 2"表示训练的时候选用两块GPU，优先选用"1"号GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pywt
from tsmoothie.smoother import *

### USE KALMAN FILTER TO SMOOTH ALL DATA (ONLY VISUALIZATION PURPOSE) ###

smoother = KalmanSmoother(component='level_longseason',
                          component_noise={'level':0.1, 'longseason':0.1},
                          n_longseasons=365)

csvdata = read_csv('London_hh_residential.csv', engine='python').values
original_data = read_csv('London_hh_residential.csv', engine='python').values

def wavelet_denoising(data, coe=0.0):
    #np.random_seed(6)
    # 小波函数取db4
    db4 = pywt.Wavelet('db4')
    # 分解
    coeffs = pywt.wavedec(data, db4)
    # 高频系数置零
    coeffs[len(coeffs) - 1] *= coe
    #coeffs[len(coeffs) - 2] *= 0
    # 重构
    meta = pywt.waverec(coeffs, db4)
    #print(data)
    #print(meta)
    return meta

def add_gaussian_noise(tensor, mean=0, std=2.5):
    #print(tensor.shape)
    noise_p = abs(np.mean(tensor[:,0]))*2.5/100

    noise = np.random.normal(0, noise_p, tensor.shape)

    #noise = torch.randn((tensor.size)) * std
    noise[:, 1] = np.zeros((noise.shape[0]))
    noisy_tensor = tensor.copy() + noise
    return noisy_tensor

class London_sm_data(Dataset):
    def __init__(self, dataset_type='train', start_date='2013-09-01', train_split='2013-12-01', client=1,\
                 forecast_period=1, window_width=10, valid_split=0.1):
        self.dataset_type = dataset_type
        self.start_date = start_date
        self.train_split = train_split
        self.client = client
        self.forecast_period = forecast_period
        self.window_width = window_width
        self.valid_split = valid_split
        self.train_data = []
        self.test_data = []
        random.seed(6)
        london_data = csvdata
        #original_data = csvdata.copy()
        #original_data = copy.deepcopy(csvdata)
        # locate the desired load data
        start_index = 0
        while london_data[start_index, 0] != 'client_' + str(self.client):
            start_index += 1
        while london_data[start_index, 1] != self.start_date:
            start_index += 1
        end_index = start_index
        train_index = start_index
        while london_data[train_index, 1] != self.train_split:
            train_index += 1
        while london_data[end_index, 0] == 'client_' + str(self.client):
            end_index += 1
        #std_scaler = lambda x: (x - np.mean(x)) / np.std(x)
        #max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        #london_data[start_index:end_index, 2] = (london_data[start_index:end_index, 2]-min(london_data[start_index:end_index, 2]))\
        #/(max(london_data[start_index:end_index, 2])-min(london_data[start_index:end_index, 2]))
        #london_data[start_index:end_index, 2] = (london_data[start_index:end_index, 2] - np.mean(
        #london_data[start_index:end_index, 2])) /np.std(london_data[start_index:end_index, 2])
        if self.client < 5:
            london_data[start_index:end_index, 2] = wavelet_denoising(london_data[start_index:end_index, 2], coe=0.025)
        else:
            london_data[start_index:end_index, 2] = wavelet_denoising(london_data[start_index:end_index, 2], coe=0)
        #print(london_data[start_index:end_index, 2])
        while start_index + self.window_width + self.forecast_period <= train_index:
            statistical_f = []
            statistical_f += london_data[start_index + self.window_width, 4:].tolist()
            #load_seq = torch.cat((torch.FloatTensor(london_data[start_index:start_index + self.window_width, 2:4]
            #                                          .astype('float32').tolist()), \
            #                      torch.FloatTensor(london_data[start_index-96:start_index-96 + self.window_width, 2:4]
            #                                        .astype('float32').tolist())), dim=1)
            #n_std = 0
            #if self.client < 5:
            #    n_std = 0
            #disturbed_load = add_gaussian_noise(london_data[start_index:start_index + self.window_width, 2:4]
            #                                          .astype('float32'), std=n_std)
            disturbed_load = london_data[start_index:start_index + self.window_width, 2:4].astype('float32')
            load_seq = torch.FloatTensor(disturbed_load.tolist())
            #load_seq = load_seq[]
            self.train_data.append((load_seq, torch.FloatTensor(statistical_f),
                               torch.FloatTensor(london_data[start_index + self.window_width:
                                                             start_index + self.window_width + self.forecast_period, 2]
                                                 .astype('float32').tolist())))
            start_index += self.forecast_period
        random.shuffle(self.train_data)

        if self.dataset_type == 'train':
            self.len = len(self.train_data[:int(len(self.train_data) * (1-self.valid_split))])

        if self.dataset_type == 'valid':
            self.len = len(self.train_data[int(len(self.train_data) * (1-self.valid_split)):])

        if self.dataset_type == 'test':
            while train_index + self.window_width + self.forecast_period <= end_index:
                statistical_f = []
                statistical_f += london_data[train_index + self.window_width, 4:].tolist()
                #load_seq = torch.cat((torch.FloatTensor(london_data[train_index:train_index + self.window_width, 2:4]
                #                                        .astype('float32').tolist()), \
                #                      torch.FloatTensor(london_data[train_index - 96:train_index - 96 + self.window_width, 2:4]
                #                          .astype('float32').tolist())), dim=1)
                load_seq = torch.FloatTensor(london_data[train_index:train_index + self.window_width, 2:4]
                                             .astype('float32').tolist())
                self.test_data.append((load_seq,
                                       torch.FloatTensor(statistical_f),
                                      torch.FloatTensor(original_data[train_index + self.window_width:
                                                                    train_index + self.window_width + self.forecast_period,
                                                        2].astype('float32').tolist())))
                train_index += self.forecast_period
            self.len = len(self.test_data)

    def __getitem__(self, index):
        if self.dataset_type == 'train':
            return self.train_data[:int(len(self.train_data) * (1-self.valid_split))][index][0], \
                   self.train_data[:int(len(self.train_data) * (1-self.valid_split))][index][1], \
                   self.train_data[:int(len(self.train_data) * (1-self.valid_split))][index][2]

        if self.dataset_type == 'valid':
            return self.train_data[int(len(self.train_data) * (1-self.valid_split)):][index][0], \
                   self.train_data[int(len(self.train_data) * (1-self.valid_split)):][index][1], \
                   self.train_data[int(len(self.train_data) * (1-self.valid_split)):][index][2]

        if self.dataset_type == 'test':
            return self.test_data[index][0], self.test_data[index][1], self.test_data[index][2]

    def get_test_data(self):
        #testx = [self.test_data[i][0] for i in range(self.len)]
        #testy = [self.test_data[i][1] for i in range(self.len)]
        return [self.test_data[k][0] for k in range(len(self.test_data))], \
               [self.test_data[k][1] for k in range(len(self.test_data))], \
               [self.test_data[k][2] for k in range(len(self.test_data))]

    def __len__(self):
        return self.len

#ds = London_sm_data(client=1, dataset_type='test')
#testx, sf, testy = ds.get_test_data()
#print(testx)
#print(testy)
#trainx = ds.__getitem__(0)
#print(trainx)
#print(ds.__len__())


