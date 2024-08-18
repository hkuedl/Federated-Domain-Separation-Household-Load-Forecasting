import pandas as pd
#import csv
#from pandas import read_csv
import numpy as np
import os

#"1, 2"表示训练的时候选用两块GPU，优先选用"1"号GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import torch
import os
import argparse
from Forecasting_Models import init_global_mapping
from server import Fedavg_server, individual, only_Fed, local_fine_tune
from client import client_group
from Forecasting_Models import mmd, diff_loss, global_mapping, local_align_mapping, local_orthogonal_mapping,\
    Fed_avg, init_global_mapping, mae

from experimental_parameters import args

#myclients = client_group(args['client_num'], torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

#print()

def test_fedavg(myclients, dev=args['device'],
                  embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1, window_width=24):

    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    # build the models
    # to avoid too many models being sent to the device, only parameters are updated and saved in training phase
    Fed_avg_model = Fed_avg(input_size, feature_size, embedding_size, hidden_size,
                            num_layers, forecast_period, window_width=window_width).to(dev)

    for client_per in args['client_per_space']:
        for alpha in args['alpha_searching_space']:
            for beta in args['beta_searching_space']:
                val_loss = 0
                test_mae = 0
                test_mse = 0
                test_mape = 0
                best_mae, best_mse, best_mape = 100, 100, 100
                worst_mae, worst_mse, worst_mape = 0, 0, 0
                print('client_per:', client_per, 'alpha:', alpha, 'beta:', beta)
                for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
                    Fed_avg_model.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                               format(clients, client_per, alpha, beta)))
                    mae, mse, mape = myclients.clients_set[clients].local_test(Fed_avg_model, if_global_mapping=True)
                    test_mae += mae
                    test_mse += mse
                    test_mape += mape
                    if best_mae > mae:
                        best_mae = mae
                    if best_mse > mse:
                        best_mse = mse
                    if best_mape > mape:
                        best_mape = mape
                    if worst_mae < mae:
                        worst_mae = mae
                    if worst_mse < mse:
                        worst_mse = mse
                    if worst_mape < mape:
                        worst_mape = mape
                    val_loss += myclients.clients_set[clients].local_val(Fed_avg_model, alpha, beta,
                                                                         if_local_mapping=True)[0]

                test_mae /= myclients.num_of_clients
                test_mse /= myclients.num_of_clients
                test_mape /= myclients.num_of_clients
                val_loss /= myclients.num_of_clients

                print('val_loss:', val_loss)
                print('{'+'%.2f' % (test_mae * 1000)+'}', end=' &')
                print('{'+'%.2f' % (best_mae * 1000)+'}', end=' &')
                print('{'+'%.2f' % (worst_mae * 1000)+'}', end=' &')
                print('{'+'%.2f' % (test_mse * 1000)+'}', end=' &')
                print('{'+'%.2f' % (best_mse * 1000)+'}', end=' &')
                print('{'+'%.2f' % (worst_mse * 1000)+'}', end=' &')
                print('{'+'%.2f' % (test_mape * 100)+'}', end=' &')
                print('{'+'%.2f' % (best_mape * 100)+'}', end=' &')
                print('{'+'%.2f' % (worst_mape * 100)+'}', end=' \\'+'\\')
                print(' ')

def test_individual(client_set, dev=args['device'],
                     embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1):
    print('individual start')

    myclients = client_set
    #init_global_mapping_model = init_global_mapping(input_size, feature_size, embedding_size,
    #                                                hidden_size, num_layers, forecast_period).to(dev)
    val_loss = 0
    test_mae = 0
    test_mse = 0
    test_mape = 0
    best_mae, best_mse, best_mape = 100, 100, 100
    worst_mae, worst_mse, worst_mape = 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        #init_global_mapping_model.load_state_dict(torch.load('model_parameters_storage/individual/individual_model_{}'.
        #                                         format(clients)))
        init_global_mapping_model = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                               format(clients)).to(dev)
        mae, mse, mape = myclients.clients_set[clients].local_test(init_global_mapping_model, if_global_mapping=False, title='individual')
        test_mae += mae
        test_mse += mse
        test_mape += mape
        if best_mae > mae:
            best_mae = mae
        if best_mse > mse:
            best_mse = mse
        if best_mape > mape:
            best_mape = mape
        if worst_mae < mae:
            worst_mae = mae
        if worst_mse < mse:
            worst_mse = mse
        if worst_mape < mape:
            worst_mape = mape
        val_loss += myclients.clients_set[clients].local_val(init_global_mapping_model, 0, 0,
                                                             if_local_mapping=False)
    test_mae /= myclients.num_of_clients
    test_mse /= myclients.num_of_clients
    test_mape /= myclients.num_of_clients
    val_loss /= myclients.num_of_clients

    print('val_loss:', val_loss)
    print('{' + '%.2f' % (test_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (best_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (worst_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (test_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (best_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (worst_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (test_mape * 100) + '}', end=' &')
    print('{' + '%.2f' % (best_mape * 100) + '}', end=' &')
    print('{' + '%.2f' % (worst_mape * 100) + '}', end=' \\'+'\\')
    print(' ')

def test_local_fine_tune(client_set, dev=args['device'],
                     embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1):
    print('local_fine_tune start')

    myclients = client_set

    for client_per in args['client_per_space']:
        val_loss = 0
        test_mae = 0
        test_mse = 0
        test_mape = 0
        best_mae, best_mse, best_mape = 100, 100, 100
        worst_mae, worst_mse, worst_mape = 0, 0, 0
        print('client_per:', client_per)
        for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
            #init_global_mapping_model.load_state_dict(
            #    torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
            #               format(clients, client_per)))
            init_global_mapping_model = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                                   format(clients, client_per)).to(dev)
            mae, mse, mape = myclients.clients_set[clients].local_test(init_global_mapping_model, if_global_mapping=False)
            test_mae += mae
            test_mse += mse
            test_mape += mape
            if best_mae > mae:
                best_mae = mae
            if best_mse > mse:
                best_mse = mse
            if best_mape > mape:
                best_mape = mape
            if worst_mae < mae:
                worst_mae = mae
            if worst_mse < mse:
                worst_mse = mse
            if worst_mape < mape:
                worst_mape = mape
            val_loss += myclients.clients_set[clients].local_val(init_global_mapping_model, 0, 0,
                                                                 if_local_mapping=False)
        test_mae /= myclients.num_of_clients
        test_mse /= myclients.num_of_clients
        test_mape /= myclients.num_of_clients
        val_loss /= myclients.num_of_clients

        print('val_loss:', val_loss)
        print('{' + '%.2f' % (test_mae * 1000) + '}', end=' &')
        print('{' + '%.2f' % (best_mae * 1000) + '}', end=' &')
        print('{' + '%.2f' % (worst_mae * 1000) + '}', end=' &')
        print('{' + '%.2f' % (test_mse * 1000) + '}', end=' &')
        print('{' + '%.2f' % (best_mse * 1000) + '}', end=' &')
        print('{' + '%.2f' % (worst_mse * 1000) + '}', end=' &')
        print('{' + '%.2f' % (test_mape * 100) + '}', end=' &')
        print('{' + '%.2f' % (best_mape * 100) + '}', end=' &')
        print('{' + '%.2f' % (worst_mape * 100) + '}', end=' \\'+'\\')
        print(' ')

def test_only_fed(client_set, dev=args['device'],
                     embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1):
    print('only_fed start')

    myclients = client_set

    for client_per in args['client_per_space']:
        val_loss = 0
        test_mae = 0
        test_mse = 0
        test_mape = 0
        best_mae, best_mse, best_mape = 100, 100, 100
        worst_mae, worst_mse, worst_mape = 0, 0, 0
        print('client_per:', client_per)
        for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
            #init_global_mapping_model.load_state_dict(
            #    torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
            #               format(clients, client_per)))
            init_global_mapping_model = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                                   format(client_per)).to(dev)
            mae, mse, mape = myclients.clients_set[clients].local_test(init_global_mapping_model, if_global_mapping=False)
            test_mae += mae
            test_mse += mse
            test_mape += mape
            if best_mae > mae:
                best_mae = mae
            if best_mse > mse:
                best_mse = mse
            if best_mape > mape:
                best_mape = mape
            if worst_mae < mae:
                worst_mae = mae
            if worst_mse < mse:
                worst_mse = mse
            if worst_mape < mape:
                worst_mape = mape
            val_loss += myclients.clients_set[clients].local_val(init_global_mapping_model, 0, 0,
                                                                 if_local_mapping=False)
        test_mae /= myclients.num_of_clients
        test_mse /= myclients.num_of_clients
        test_mape /= myclients.num_of_clients
        val_loss /= myclients.num_of_clients

        print('val_loss:', val_loss)
        print('{' + '%.2f' % (test_mae * 1000) + '}', end=' &')
        print('{' + '%.2f' % (best_mae * 1000) + '}', end=' &')
        print('{' + '%.2f' % (worst_mae * 1000) + '}', end=' &')
        print('{' + '%.2f' % (test_mse * 1000) + '}', end=' &')
        print('{' + '%.2f' % (best_mse * 1000) + '}', end=' &')
        print('{' + '%.2f' % (worst_mse * 1000) + '}', end=' &')
        print('{' + '%.2f' % (test_mape * 100) + '}', end=' &')
        print('{' + '%.2f' % (best_mape * 100) + '}', end=' &')
        print('{' + '%.2f' % (worst_mape * 100) + '}', end=' \\'+'\\')
        print(' ')

def test_local_persist(client_set):
    print('persist start')

    myclients = client_set
    #init_global_mapping_model = init_global_mapping(input_size, feature_size, embedding_size,
    #                                                hidden_size, num_layers, forecast_period).to(dev)
    val_loss = 0
    test_mae = 0
    test_mse = 0
    test_mape = 0
    best_mae, best_mse, best_mape = 100, 100, 100
    worst_mae, worst_mse, worst_mape = 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        #init_global_mapping_model.load_state_dict(torch.load('model_parameters_storage/individual/individual_model_{}'.
        #                                         format(clients)))
        mae, mse, mape = myclients.clients_set[clients].local_persist()
        test_mae += mae
        test_mse += mse
        test_mape += mape
        if best_mae > mae:
            best_mae = mae
        if best_mse > mse:
            best_mse = mse
        if best_mape > mape:
            best_mape = mape
        if worst_mae < mae:
            worst_mae = mae
        if worst_mse < mse:
            worst_mse = mse
        if worst_mape < mape:
            worst_mape = mape

    test_mae /= myclients.num_of_clients
    test_mse /= myclients.num_of_clients
    test_mape /= myclients.num_of_clients
    val_loss /= myclients.num_of_clients

    print('val_loss:', val_loss)
    print('{' + '%.2f' % (test_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (best_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (worst_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (test_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (best_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (worst_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (test_mape * 100) + '}', end=' &')
    print('{' + '%.2f' % (best_mape * 100) + '}', end=' &')
    print('{' + '%.2f' % (worst_mape * 100) + '}', end=' \\'+'\\')
    print(' ')

def test_nbeats(client_set, dev=args['device'],
                     embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1):
    print('individual start')

    myclients = client_set
    #init_global_mapping_model = init_global_mapping(input_size, feature_size, embedding_size,
    #                                                hidden_size, num_layers, forecast_period).to(dev)
    val_loss = 0
    test_mae = 0
    test_mse = 0
    test_mape = 0
    best_mae, best_mse, best_mape = 100, 100, 100
    worst_mae, worst_mse, worst_mape = 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        #init_global_mapping_model.load_state_dict(torch.load('model_parameters_storage/individual/individual_model_{}'.
        #                                         format(clients)))
        init_global_mapping_model = torch.load('model_parameters_storage/nbeats/nbeats_model_{}'.
                                               format(clients)).to(dev)
        mae, mse, mape = myclients.clients_set[clients].local_test(init_global_mapping_model, if_global_mapping=False, title='individual')
        test_mae += mae
        test_mse += mse
        test_mape += mape
        if best_mae > mae:
            best_mae = mae
        if best_mse > mse:
            best_mse = mse
        if best_mape > mape:
            best_mape = mape
        if worst_mae < mae:
            worst_mae = mae
        if worst_mse < mse:
            worst_mse = mse
        if worst_mape < mape:
            worst_mape = mape
        val_loss += myclients.clients_set[clients].local_val(init_global_mapping_model, 0, 0,
                                                             if_local_mapping=False)
    test_mae /= myclients.num_of_clients
    test_mse /= myclients.num_of_clients
    test_mape /= myclients.num_of_clients
    val_loss /= myclients.num_of_clients

    print('val_loss:', val_loss)
    print('{' + '%.2f' % (test_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (best_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (worst_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (test_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (best_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (worst_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (test_mape * 100) + '}', end=' &')
    print('{' + '%.2f' % (best_mape * 100) + '}', end=' &')
    print('{' + '%.2f' % (worst_mape * 100) + '}', end=' \\'+'\\')
    print(' ')

def test_impactnet(client_set, dev=args['device'],
                     embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1):
    print('individual start')

    myclients = client_set
    #init_global_mapping_model = init_global_mapping(input_size, feature_size, embedding_size,
    #                                                hidden_size, num_layers, forecast_period).to(dev)
    val_loss = 0
    test_mae = 0
    test_mse = 0
    test_mape = 0
    best_mae, best_mse, best_mape = 100, 100, 100
    worst_mae, worst_mse, worst_mape = 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        #init_global_mapping_model.load_state_dict(torch.load('model_parameters_storage/individual/individual_model_{}'.
        #                                         format(clients)))
        init_global_mapping_model = torch.load('model_parameters_storage/impactnet/impactnet_model_{}'.
                                               format(clients)).to(dev)
        mae, mse, mape = myclients.clients_set[clients].local_test(init_global_mapping_model, if_global_mapping=False, title='individual')
        test_mae += mae
        test_mse += mse
        test_mape += mape
        if best_mae > mae:
            best_mae = mae
        if best_mse > mse:
            best_mse = mse
        if best_mape > mape:
            best_mape = mape
        if worst_mae < mae:
            worst_mae = mae
        if worst_mse < mse:
            worst_mse = mse
        if worst_mape < mape:
            worst_mape = mape
        val_loss += myclients.clients_set[clients].local_val(init_global_mapping_model, 0, 0,
                                                             if_local_mapping=False)
    test_mae /= myclients.num_of_clients
    test_mse /= myclients.num_of_clients
    test_mape /= myclients.num_of_clients
    val_loss /= myclients.num_of_clients

    print('val_loss:', val_loss)
    print('{' + '%.2f' % (test_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (best_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (worst_mae * 1000) + '}', end=' &')
    print('{' + '%.2f' % (test_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (best_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (worst_mse * 1000) + '}', end=' &')
    print('{' + '%.2f' % (test_mape * 100) + '}', end=' &')
    print('{' + '%.2f' % (best_mape * 100) + '}', end=' &')
    print('{' + '%.2f' % (worst_mape * 100) + '}', end=' \\'+'\\')
    print(' ')

#test_fedavg(myclients=myclients)
#test_individual()
#test_local_fine_tune()
#test_only_fed()