import pandas as pd
#import csv
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import os

#"1, 2"表示训练的时候选用两块GPU，优先选用"1"号GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import torch
import argparse
from Forecasting_Models import init_global_mapping
from server import Fedavg_server, individual, only_Fed, local_fine_tune
from client import client_group
from Forecasting_Models import mmd, diff_loss, global_mapping, local_align_mapping, local_orthogonal_mapping,\
    Fed_avg, init_global_mapping, mae, Fed_avg_without_personalization, NBEATS, impactnet


from experimental_parameters import args

myclients = client_group(args['client_num'], args['device'],
                         window_width=args['window_width'])

def read_mae_com1(dev=myclients.dev, client_per=0.3, alpha=1, beta=1):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)

    num_client=1
    av_mae1, av_mae2, av_mae3, av_mae4, av_mae5 = 0, 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:

        Fed_avg_model.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                 format(clients, client_per, alpha, beta)))
        individual_model = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                               format(clients)).to(dev)
        local_ft_model = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                               format(clients, client_per)).to(dev)
        only_Fed_model = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                               format(client_per)).to(dev)
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model, if_global_mapping=True)[0]
        mae2 = myclients.clients_set[clients].local_persist()[0]
        mae3 = myclients.clients_set[clients].local_test(individual_model, if_global_mapping=False)[0]
        mae4 = myclients.clients_set[clients].local_test(only_Fed_model, if_global_mapping=False)[0]
        mae5 = myclients.clients_set[clients].local_test(local_ft_model, if_global_mapping=False)[0]

        av_mae1 += mae1
        av_mae2 += mae2
        av_mae3 += mae3
        av_mae4 += mae4
        av_mae5 += mae5

        print('\#'+str(num_client), end=' &')
        print('{' + '%.2f' % (mae1 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae2 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae5 * 1000) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    av_mae1 /= myclients.num_of_clients
    av_mae2 /= myclients.num_of_clients
    av_mae3 /= myclients.num_of_clients
    av_mae4 /= myclients.num_of_clients
    av_mae5 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (av_mae1 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae2 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae3 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae4 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae5 * 1000) + '}', end=' \\' + '\\')
    print(' ')

#read_mae_com1(client_per=0.5)

def read_rmse_com1(dev=myclients.dev, client_per=0.3, alpha=1, beta=1):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)

    num_client=1
    av_mae1, av_mae2, av_mae3, av_mae4, av_mae5 = 0, 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:

        Fed_avg_model.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                 format(clients, client_per, alpha, beta)))
        individual_model = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                               format(clients)).to(dev)
        local_ft_model = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                               format(clients, client_per)).to(dev)
        only_Fed_model = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                               format(client_per)).to(dev)
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model, if_global_mapping=True)[1]
        mae2 = myclients.clients_set[clients].local_persist()[1]
        mae3 = myclients.clients_set[clients].local_test(individual_model, if_global_mapping=False)[1]
        mae4 = myclients.clients_set[clients].local_test(only_Fed_model, if_global_mapping=False)[1]
        mae5 = myclients.clients_set[clients].local_test(local_ft_model, if_global_mapping=False)[1]

        av_mae1 += mae1
        av_mae2 += mae2
        av_mae3 += mae3
        av_mae4 += mae4
        av_mae5 += mae5

        print('\#'+str(num_client), end=' &')
        print('{' + '%.2f' % (mae1 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae2 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae5 * 1000) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    av_mae1 /= myclients.num_of_clients
    av_mae2 /= myclients.num_of_clients
    av_mae3 /= myclients.num_of_clients
    av_mae4 /= myclients.num_of_clients
    av_mae5 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (av_mae1 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae2 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae3 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae4 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae5 * 1000) + '}', end=' \\' + '\\')
    print(' ')

#read_rmse_com1(client_per=0.5)

def read_mape_com1(dev=myclients.dev, client_per=0.3, alpha=1, beta=1):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)

    num_client=1
    av_mae1, av_mae2, av_mae3, av_mae4, av_mae5 = 0, 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:

        Fed_avg_model.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                 format(clients, client_per, alpha, beta)))
        individual_model = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                               format(clients)).to(dev)
        local_ft_model = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                               format(clients, client_per)).to(dev)
        only_Fed_model = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                               format(client_per)).to(dev)
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model, if_global_mapping=True)[2]
        mae2 = myclients.clients_set[clients].local_persist()[2]
        mae3 = myclients.clients_set[clients].local_test(individual_model, if_global_mapping=False)[2]
        mae4 = myclients.clients_set[clients].local_test(only_Fed_model, if_global_mapping=False)[2]
        mae5 = myclients.clients_set[clients].local_test(local_ft_model, if_global_mapping=False)[2]

        av_mae1 += mae1
        av_mae2 += mae2
        av_mae3 += mae3
        av_mae4 += mae4
        av_mae5 += mae5

        print('\#'+str(num_client), end=' &')
        print('{' + '%.2f' % (mae1 * 100) + '}', end=' &')
        print('{' + '%.2f' % (mae2 * 100) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 100) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 100) + '}', end=' &')
        print('{' + '%.2f' % (mae5 * 100) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    av_mae1 /= myclients.num_of_clients
    av_mae2 /= myclients.num_of_clients
    av_mae3 /= myclients.num_of_clients
    av_mae4 /= myclients.num_of_clients
    av_mae5 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (av_mae1 * 100) + '}', end=' &')
    print('{' + '%.2f' % (av_mae2 * 100) + '}', end=' &')
    print('{' + '%.2f' % (av_mae3 * 100) + '}', end=' &')
    print('{' + '%.2f' % (av_mae4 * 100) + '}', end=' &')
    print('{' + '%.2f' % (av_mae5 * 100) + '}', end=' \\' + '\\')
    print(' ')

#read_mape_com1(client_per=0.5)

def plot_com1(dev=myclients.dev, client_per=0.3, alpha=1, beta=1, clients='client0', xlim = 0, tlim = 0.6, zoomx1 = 254, max_y=1.6,
              xlim2=0, tlim2=0.6, zoomx3=30):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)

    num_client=1

    Fed_avg_model.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                             format(clients, client_per, alpha, beta)))
    individual_model = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                  format(clients)).to(dev)
    local_ft_model = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                format(clients, client_per)).to(dev)
    only_Fed_model = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                format(client_per)).to(dev)
    true, mae1 = myclients.clients_set[clients].local_test(Fed_avg_model, if_global_mapping=True, if_return_list=True)
    mae3 = myclients.clients_set[clients].local_test(individual_model, if_global_mapping=False, if_return_list=True)[1]
    mae4 = myclients.clients_set[clients].local_test(only_Fed_model, if_global_mapping=False, if_return_list=True)[1]
    mae5 = myclients.clients_set[clients].local_test(local_ft_model, if_global_mapping=False, if_return_list=True)[1]

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))
    ax.plot(true[:720], label='True', color='black', ls='--', lw=1.5)
    ax.plot(range(1,720), true[:719], label='Method 1', color='orange', ls='-', lw=1)
    ax.plot(mae3[:720], label='Method 2', color='royalblue', ls='-', lw=1)
    ax.plot(mae4[:720], label='Method 3', color='darkcyan', ls='-', lw=1)
    ax.plot(mae5[:720], label='Method 4', color='cyan', ls='-', lw=1)
    ax.plot(mae1[:720], label='Proposed', color='crimson', ls='-', lw=1)
    ax.set_xlabel('Time index [hour]', fontproperties='Times New Roman')
    ax.set_ylabel('Electrical load [kWh]', fontproperties='Times New Roman')
    plt.margins(x=0)
    ax.set_ylim(0, max_y)

    # zoom1
    zoomx2 = zoomx1+48
    axins_1 = ax.inset_axes((0.5, 0.58, 0.2, 0.3))
    axins_1.plot(true[:720], color='black', ls='--', lw=1)
    axins_1.plot(range(1, 720), true[:719], color='orange', ls='-', lw=1)
    axins_1.plot(mae3[:720], color='royalblue', ls='-', lw=1)
    axins_1.plot(mae4[:720], color='darkcyan', ls='-', lw=1)
    axins_1.plot(mae5[:720], color='cyan', ls='-', lw=1)
    axins_1.plot(mae1[:720], color='crimson', ls='-', lw=1)
    axins_1.set_xlim(zoomx1, zoomx2)
    axins_1.set_ylim(xlim, tlim)
    axins_1.set_title('Zoom in', fontproperties='Times New Roman')
    tx0 = zoomx1
    tx1 = zoomx2
    ty0 = xlim
    ty1 = tlim
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty0)
    xy2 = (tx0, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)
    xy = (tx1, ty0)
    xy2 = (tx1, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)

    # zoom1
    zoomx4 = zoomx3 + 48
    axins_2 = ax.inset_axes((0.1, 0.58, 0.2, 0.3))
    axins_2.plot(true[:720], color='black', ls='--', lw=1)
    axins_2.plot(range(1, 720), true[:719], color='orange', ls='-', lw=1)
    axins_2.plot(mae3[:720], color='royalblue', ls='-', lw=1)
    axins_2.plot(mae4[:720], color='darkcyan', ls='-', lw=1)
    axins_2.plot(mae5[:720], color='cyan', ls='-', lw=1)
    axins_2.plot(mae1[:720], color='crimson', ls='-', lw=1)
    axins_2.set_xlim(zoomx3, zoomx4)
    axins_2.set_ylim(xlim2, tlim2)
    axins_2.set_title('Zoom in', fontproperties='Times New Roman')
    tx0 = zoomx3
    tx1 = zoomx4
    ty0 = xlim2
    ty1 = tlim2
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty0)
    xy2 = (tx0, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_2, axesB=ax)
    con.set_color('silver')
    axins_2.add_artist(con)
    xy = (tx1, ty0)
    xy2 = (tx1, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_2, axesB=ax)
    con.set_color('silver')
    axins_2.add_artist(con)

    plt.setp(axins_2.get_xticklabels(), fontproperties='Times New Roman')
    plt.setp(axins_2.get_yticklabels(), fontproperties='Times New Roman')

    plt.setp(axins_1.get_xticklabels(), fontproperties='Times New Roman')
    plt.setp(axins_1.get_yticklabels(), fontproperties='Times New Roman')
    plt.setp(ax.get_xticklabels(), fontproperties='Times New Roman')
    plt.setp(ax.get_yticklabels(), fontproperties='Times New Roman')

    font = {'family': 'Times New Roman'}
    plt.legend(loc='upper right', prop=font, edgecolor='black', shadow=False, fancybox=False, labelspacing=0.15)
    plt.savefig(fname='Figs/comp_1_'+clients+'.eps', format='eps', bbox_inches='tight')
    plt.show()

#plot_com1(client_per=0.5, clients='client0')
#plot_com1(client_per=0.5, clients='client1', xlim=0, tlim=1.1, max_y=3)
#plot_com1(client_per=0.5, clients='client7', xlim=0, tlim=1.2, max_y=3)
#plot_com1(client_per=0.5, clients='client0', zoomx1 = 359, tlim=0.6)
#plot_com1(client_per=0.5, clients='client1', xlim = 0.08, tlim = 0.5, zoomx1 = 591, max_y=3, zoomx3=320,xlim2=0.08, tlim2=0.5)

def read_mae_com2(dev=myclients.dev, client_per=0.3):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model1 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model2 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model3 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model4 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)

    num_client=1
    av_mae1, av_mae3, av_mae4, av_mae5 = 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:

        Fed_avg_model1.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                 format(clients, client_per, 1, 1)))
        Fed_avg_model2.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 0)))
        Fed_avg_model3.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 1)))
        Fed_avg_model4.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 0)))
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model1, if_global_mapping=True)[0]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=True)[0]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=True)[0]
        mae5 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=True)[0]

        av_mae1 += mae1
        av_mae3 += mae3
        av_mae4 += mae4
        av_mae5 += mae5

        print('\#'+str(num_client), end=' &')
        print('{' + '%.2f' % (mae1 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae5 * 1000) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    av_mae1 /= myclients.num_of_clients
    av_mae3 /= myclients.num_of_clients
    av_mae4 /= myclients.num_of_clients
    av_mae5 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (av_mae1 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae3 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae4 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae5 * 1000) + '}', end=' \\' + '\\')
    print(' ')

#read_mae_com2(client_per=0.5)

def read_rmse_com2(dev=myclients.dev, client_per=0.3):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model1 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model2 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model3 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model4 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)

    num_client=1
    av_mae1, av_mae3, av_mae4, av_mae5 = 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:

        Fed_avg_model1.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                 format(clients, client_per, 1, 1)))
        Fed_avg_model2.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 0)))
        Fed_avg_model3.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 1)))
        Fed_avg_model4.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 0)))
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model1, if_global_mapping=True)[1]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=True)[1]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=True)[1]
        mae5 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=True)[1]

        av_mae1 += mae1
        av_mae3 += mae3
        av_mae4 += mae4
        av_mae5 += mae5

        print('\#'+str(num_client), end=' &')
        print('{' + '%.2f' % (mae1 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae5 * 1000) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    av_mae1 /= myclients.num_of_clients
    av_mae3 /= myclients.num_of_clients
    av_mae4 /= myclients.num_of_clients
    av_mae5 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (av_mae1 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae3 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae4 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae5 * 1000) + '}', end=' \\' + '\\')
    print(' ')

#read_rmse_com2(client_per=0.5)

def read_mape_com2(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model1 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model2 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model3 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model4 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)

    num_client=1
    av_mae1, av_mae3, av_mae4, av_mae5 = 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:

        Fed_avg_model1.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                 format(clients, client_per, 1, 1)))
        Fed_avg_model2.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 0)))
        Fed_avg_model3.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 1)))
        Fed_avg_model4.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 0)))
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model1, if_global_mapping=True)[2]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=True)[2]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=True)[2]
        mae5 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=True)[2]

        av_mae1 += mae1
        av_mae3 += mae3
        av_mae4 += mae4
        av_mae5 += mae5

        print('\#'+str(num_client), end=' &')
        print('{' + '%.2f' % (mae1 * 100) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 100) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 100) + '}', end=' &')
        print('{' + '%.2f' % (mae5 * 100) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    av_mae1 /= myclients.num_of_clients
    av_mae3 /= myclients.num_of_clients
    av_mae4 /= myclients.num_of_clients
    av_mae5 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (av_mae1 * 100) + '}', end=' &')
    print('{' + '%.2f' % (av_mae3 * 100) + '}', end=' &')
    print('{' + '%.2f' % (av_mae4 * 100) + '}', end=' &')
    print('{' + '%.2f' % (av_mae5 * 100) + '}', end=' \\' + '\\')
    print(' ')

#read_mape_com2(client_per=0.5)

def plot_com2(dev=myclients.dev, client_per=0.3, alpha=1, beta=1, clients='client0', xlim = 0.08, tlim = 1.8, zoomx1 = 281):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model1 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model2 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model3 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model4 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)

    num_client=1

    Fed_avg_model1.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                              format(clients, client_per, 1, 1)))
    Fed_avg_model2.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                              format(clients, client_per, 0, 0)))
    Fed_avg_model3.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                              format(clients, client_per, 0, 1)))
    Fed_avg_model4.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                              format(clients, client_per, 1, 0)))
    true, mae1 = myclients.clients_set[clients].local_test(Fed_avg_model1, if_global_mapping=True, if_return_list=True)
    mae3 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=True, if_return_list=True)[1]
    mae4 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=True, if_return_list=True)[1]
    mae5 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=True, if_return_list=True)[1]

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(true[:720], label='True', color='black', ls='--', lw=1.5)
    ax.plot(mae3[:720], label='Method 5', color='royalblue', ls='-', lw=1.5)
    ax.plot(mae4[:720], label='Method 6', color='orange', ls='-', lw=1.5)
    ax.plot(mae5[:720], label='Method 7', color='cyan', ls='-', lw=1.5)
    ax.plot(mae1[:720], label='Proposed', color='crimson', ls='-', lw=1.5)
    ax.set_xlabel('Time index [hour]', fontproperties='Times New Roman')
    ax.set_ylabel('Electrical load [MW]', fontproperties='Times New Roman')
    plt.margins(x=0)
    ax.set_ylim(0, 3)

    # zoom1
    zoomx2 = zoomx1 + 48
    axins_1 = ax.inset_axes((0.45, 0.6, 0.5, 0.3))
    axins_1.plot(true[:720], color='black', ls='--', lw=1.5)
    axins_1.plot(mae3[:720], color='royalblue', ls='-', lw=1.5)
    axins_1.plot(mae4[:720], color='orange', ls='-', lw=1.5)
    axins_1.plot(mae5[:720], color='cyan', ls='-', lw=1.5)
    axins_1.plot(mae1[:720], color='crimson', ls='-', lw=1.5)
    axins_1.set_xlim(zoomx1, zoomx2)
    axins_1.set_ylim(xlim, tlim)
    axins_1.set_title('Zoom in', fontproperties='Times New Roman')
    tx0 = zoomx1
    tx1 = zoomx2
    ty0 = xlim
    ty1 = tlim
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty0)
    xy2 = (tx0, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)
    xy = (tx1, ty0)
    xy2 = (tx1, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)

    plt.setp(axins_1.get_xticklabels(), fontproperties='Times New Roman')
    plt.setp(axins_1.get_yticklabels(), fontproperties='Times New Roman')
    plt.setp(ax.get_xticklabels(), fontproperties='Times New Roman')
    plt.setp(ax.get_yticklabels(), fontproperties='Times New Roman')

    font = {'family': 'Times New Roman'}
    plt.legend(loc='upper left', prop=font, edgecolor='black', shadow=False, fancybox=False, labelspacing=0.15)
    plt.savefig(fname='Figs/comp_2_'+clients+'.eps', format='eps', bbox_inches='tight')
    plt.show()

#plot_com2(client_per=0.5, clients='client3')
#plot_com2(client_per=0.5, clients='client14', zoomx1=181)
#plot_com2(client_per=0.5, clients='client15', zoomx1=181)
#plot_com2(client_per=0.5, clients='client16', zoomx1=181)

def plot_points_com2(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model1 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model2 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model3 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model4 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)

    num_client=1
    av_mae1, av_mae3, av_mae4, av_mae5 = [], [], [], []
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:

        Fed_avg_model1.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                 format(clients, client_per, 1, 1)))
        Fed_avg_model2.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 0)))
        Fed_avg_model3.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 1)))
        Fed_avg_model4.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 0)))
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model1, if_global_mapping=True)[0]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=True)[0]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=True)[0]
        mae5 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=True)[0]

        av_mae1.append(mae1*1000)
        av_mae3.append(mae3*1000)
        av_mae4.append(mae4*1000)
        av_mae5.append(mae5*1000)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(6,3))
    plt.scatter(av_mae1, av_mae3, marker='o', color='crimson', label='Method 5')
    plt.scatter(av_mae1, av_mae4, marker='v', color='royalblue', label='Method 6')
    plt.scatter(av_mae1, av_mae5, marker='*', color='cadetblue', label='Method 7')
    plt.plot([0,200], [0,200], ls='--', lw=1.5, color='black')

    plt.xlabel('Proposed framework', fontproperties='Times New Roman')
    plt.ylabel('Comparison methods', fontproperties='Times New Roman')
    plt.margins(x=0, y=0)

    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')

    font = {'family': 'Times New Roman'}
    plt.legend(loc='upper left', prop=font, edgecolor='black', shadow=False, fancybox=False, labelspacing=0.15)
    plt.show()

#plot_points_com2(client_per=0.5)



def read_mae_com3(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model1 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model2 = Fed_avg_without_personalization(input_size, feature_size, embedding_size=20, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)


    num_client=1
    av_mae1, av_mae3, av_mae4, av_mae5 = 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:

        Fed_avg_model1.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                 format(clients, client_per, 1, 1)))
        Fed_avg_model2.load_state_dict(torch.load('model_parameters_storage/fed_avg/no_personalization_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 1)))
        Fed_avg_model3 = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                format(clients, client_per)).to(dev)
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model1, if_global_mapping=True)[0]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=True)[0]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[0]

        av_mae1 += mae1
        av_mae3 += mae3
        av_mae4 += mae4

        print('\#'+str(num_client), end=' &')
        print('{' + '%.2f' % (mae1 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 1000) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    av_mae1 /= myclients.num_of_clients
    av_mae3 /= myclients.num_of_clients
    av_mae4 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (av_mae1 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae3 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae4 * 1000) + '}', end=' \\' + '\\')
    print(' ')

#read_mae_com3(client_per=0.5)

def read_rmse_com3(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model1 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model2 = Fed_avg_without_personalization(input_size, feature_size, embedding_size=20, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)


    num_client=1
    av_mae1, av_mae3, av_mae4, av_mae5 = 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:

        Fed_avg_model1.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                 format(clients, client_per, 1, 1)))
        Fed_avg_model2.load_state_dict(torch.load('model_parameters_storage/fed_avg/no_personalization_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 1)))
        Fed_avg_model3 = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                format(clients, client_per)).to(dev)
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model1, if_global_mapping=True)[1]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=True)[1]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[1]

        av_mae1 += mae1
        av_mae3 += mae3
        av_mae4 += mae4

        print('\#'+str(num_client), end=' &')
        print('{' + '%.2f' % (mae1 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 1000) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    av_mae1 /= myclients.num_of_clients
    av_mae3 /= myclients.num_of_clients
    av_mae4 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (av_mae1 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae3 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae4 * 1000) + '}', end=' \\' + '\\')
    print(' ')
#read_rmse_com3()

def read_error_com3(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model1 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'], num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model2 = Fed_avg_without_personalization(input_size, feature_size, embedding_size=20, hidden_size=args['hidden_size'],
                            num_layers=args['num_layers'],
                            num_labels=args['num_layers'], window_width=args['window_width']).to(dev)


    num_client=1
    av_mae1, av_mae3, av_mae4, av_mse1, av_mse3, av_mse4 = 0, 0, 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:

        Fed_avg_model1.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                 format(clients, client_per, 1, 1)))
        Fed_avg_model2.load_state_dict(torch.load('model_parameters_storage/fed_avg/no_personalization_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 1)))
        Fed_avg_model3 = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                format(clients, client_per)).to(dev)
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model1, if_global_mapping=True)[0]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=True)[0]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[0]
        mse1 = myclients.clients_set[clients].local_test(Fed_avg_model1, if_global_mapping=True)[1]
        mse3 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=True)[1]
        mse4 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[1]

        av_mae1 += mae1
        av_mae3 += mae3
        av_mae4 += mae4
        av_mse1 += mse1
        av_mse3 += mse3
        av_mse4 += mse4

        print('\#'+str(num_client), end=' &')
        print('{' + '%.2f' % (mae1 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mse1 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mse3 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mse4 * 1000) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    av_mae1 /= myclients.num_of_clients
    av_mae3 /= myclients.num_of_clients
    av_mae4 /= myclients.num_of_clients
    av_mse1 /= myclients.num_of_clients
    av_mse3 /= myclients.num_of_clients
    av_mse4 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (av_mae1 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae3 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mae4 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mse1 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mse3 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (av_mse4 * 1000) + '}', end=' \\' + '\\')
    print(' ')

#read_error_com3()

def plot_points_com3(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model1 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model2 = Fed_avg_without_personalization(input_size, feature_size, embedding_size=20,
                                                     hidden_size=args['hidden_size'],
                                                     num_layers=args['num_layers'],
                                                     num_labels=args['num_layers'],
                                                     window_width=args['window_width']).to(dev)

    num_client=1
    av_mae1, av_mae3, av_mae4 = [], [], []
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        Fed_avg_model1.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 1)))
        Fed_avg_model2.load_state_dict(
            torch.load('model_parameters_storage/fed_avg/no_personalization_model_{}_{}_{}_{}'.
                       format(clients, client_per, 1, 1)))
        Fed_avg_model3 = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                    format(clients, client_per)).to(dev)
        mae1 = myclients.clients_set[clients].local_test(Fed_avg_model1, if_global_mapping=True)[0]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=True)[0]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[0]

        av_mae1.append(mae1*1000)
        av_mae3.append(mae3*1000)
        av_mae4.append(mae4*1000)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(6,3))
    plt.scatter(av_mae1, av_mae3, marker='o', color='crimson', label='Method 8')
    plt.scatter(av_mae1, av_mae4, marker='v', color='royalblue', label='Method 2')
    plt.plot([0,175], [0,175], ls='--', lw=1.5, color='black')

    plt.xlabel('Proposed framework', fontproperties='Times New Roman')
    plt.ylabel('Comparison methods', fontproperties='Times New Roman')
    plt.margins(x=0, y=0)

    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')

    font = {'family': 'Times New Roman'}
    plt.legend(loc='upper left', prop=font, edgecolor='black', shadow=False, fancybox=False, labelspacing=0.15)
    plt.show()

#plot_points_com3()

def plot_points_com123(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model0 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model5 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model6 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model7 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model8 = Fed_avg_without_personalization(input_size, feature_size, embedding_size=20,
                                                     hidden_size=args['hidden_size'],
                                                     num_layers=args['num_layers'],
                                                     num_labels=args['num_layers'],
                                                     window_width=args['window_width']).to(dev)

    num_client=1
    av_mae0, av_mae1, av_mae2, av_mae3, av_mae4, av_mae5, av_mae6, av_mae7, av_mae8, av_mae9, av_mae10  = [], [], [], [], [], [], [], [], [], [], []
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        Fed_avg_model0.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 100)))
        Fed_avg_model2 = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                      format(clients)).to(dev)
        Fed_avg_model3 = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                    format(clients, client_per)).to(dev)
        Fed_avg_model4 = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                    format(client_per)).to(dev)
        Fed_avg_model5.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 0)))
        Fed_avg_model6.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 1)))
        Fed_avg_model7.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 0)))
        Fed_avg_model8.load_state_dict(
            torch.load('model_parameters_storage/fed_avg/no_personalization_model_{}_{}_{}_{}'.
                       format(clients, client_per, 1, 1)))
        Fed_avg_model9 = torch.load('model_parameters_storage/nbeats/nbeats_model_{}'.
                                    format(clients)).to(dev)
        Fed_avg_model10 = torch.load('model_parameters_storage/impactnet/impactnet_model_{}'.
                                    format(clients)).to(dev)

        mae0 = myclients.clients_set[clients].local_test(Fed_avg_model0, if_global_mapping=True)[0]
        mae1 = myclients.clients_set[clients].local_persist()[0]
        mae2 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=False)[0]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[0]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=False)[0]
        mae5 = myclients.clients_set[clients].local_test(Fed_avg_model5, if_global_mapping=True)[0]
        mae6 = myclients.clients_set[clients].local_test(Fed_avg_model6, if_global_mapping=True)[0]
        mae7 = myclients.clients_set[clients].local_test(Fed_avg_model7, if_global_mapping=True)[0]
        mae8 = myclients.clients_set[clients].local_test(Fed_avg_model8, if_global_mapping=True)[0]
        mae9 = myclients.clients_set[clients].local_test(Fed_avg_model9, if_global_mapping=False)[0]
        mae10 = myclients.clients_set[clients].local_test(Fed_avg_model10, if_global_mapping=False)[0]

        av_mae0.append(mae0 * 1000)
        av_mae1.append(mae1*1000)
        av_mae2.append(mae2 * 1000)
        av_mae3.append(mae3*1000)
        av_mae4.append(mae4*1000)
        av_mae5.append(mae5 * 1000)
        av_mae6.append(mae6 * 1000)
        av_mae7.append(mae7 * 1000)
        av_mae8.append(mae8 * 1000)
        av_mae9.append(mae9 * 1000)
        av_mae10.append(mae10 * 1000)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(6,3))
    plt.scatter(av_mae0, av_mae1, marker='o', color='crimson', label='Method 1', s=12)
    plt.scatter(av_mae0, av_mae2, marker='v', color='cadetblue', label='Method 2', s=12)
    plt.scatter(av_mae0, av_mae3, marker='^', color='orange', label='Method 3', s=12)
    plt.scatter(av_mae0, av_mae4, marker='1', color='royalblue', label='Method 4', s=12)
    plt.scatter(av_mae0, av_mae5, marker='2', color='cyan', label='Method 5', s=12)
    plt.scatter(av_mae0, av_mae6, marker='+', color='pink', label='Method 6', s=12)
    plt.scatter(av_mae0, av_mae7, marker='x', color='greenyellow', label='Method 7', s=12)
    plt.scatter(av_mae0, av_mae8, marker='H', color='darkgoldenrod', label='Method 8', s=12)
    plt.scatter(av_mae0, av_mae9, marker='o', color='darkorange', label='Method 9', s=12)
    plt.scatter(av_mae0, av_mae10, marker='v', color='magenta', label='Method 10', s=12)
    plt.plot([0,175], [0,175], ls='--', lw=1.5, color='black')

    plt.grid(color='silver', linestyle='-', linewidth=0.5)

    plt.xlabel('MAE [Wh] of the proposed framework', fontproperties='Times New Roman')
    plt.ylabel('MAE [Wh] of comparison methods', fontproperties='Times New Roman')
    plt.margins(x=0, y=0)

    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')

    font = {'family': 'Times New Roman'}
    plt.legend(prop=font, edgecolor='black', shadow=False, fancybox=False, labelspacing=0.15)
    plt.savefig(fname='Figs/points_comp_123_.eps', format='eps', bbox_inches='tight')
    plt.show()

#plot_points_com123(client_per=0.5)

def plot_points_rmse_com123(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model0 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model5 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model6 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model7 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model8 = Fed_avg_without_personalization(input_size, feature_size, embedding_size=20,
                                                     hidden_size=args['hidden_size'],
                                                     num_layers=args['num_layers'],
                                                     num_labels=args['num_layers'],
                                                     window_width=args['window_width']).to(dev)

    num_client = 1
    av_mae0, av_mae1, av_mae2, av_mae3, av_mae4, av_mae5, av_mae6, av_mae7, av_mae8, av_mae9, av_mae10 = [], [], [], [], [], [], [], [], [], [], []
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        Fed_avg_model0.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 100)))
        Fed_avg_model2 = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                    format(clients)).to(dev)
        Fed_avg_model3 = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                    format(clients, client_per)).to(dev)
        Fed_avg_model4 = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                    format(client_per)).to(dev)
        Fed_avg_model5.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 0)))
        Fed_avg_model6.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 1)))
        Fed_avg_model7.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 0)))
        Fed_avg_model8.load_state_dict(
            torch.load('model_parameters_storage/fed_avg/no_personalization_model_{}_{}_{}_{}'.
                       format(clients, client_per, 1, 1)))
        Fed_avg_model9 = torch.load('model_parameters_storage/nbeats/nbeats_model_{}'.
                                    format(clients)).to(dev)
        Fed_avg_model10 = torch.load('model_parameters_storage/impactnet/impactnet_model_{}'.
                                     format(clients)).to(dev)

        mae0 = myclients.clients_set[clients].local_test(Fed_avg_model0, if_global_mapping=True)[1]
        mae1 = myclients.clients_set[clients].local_persist()[1]
        mae2 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=False)[1]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[1]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=False)[1]
        mae5 = myclients.clients_set[clients].local_test(Fed_avg_model5, if_global_mapping=True)[1]
        mae6 = myclients.clients_set[clients].local_test(Fed_avg_model6, if_global_mapping=True)[1]
        mae7 = myclients.clients_set[clients].local_test(Fed_avg_model7, if_global_mapping=True)[1]
        mae8 = myclients.clients_set[clients].local_test(Fed_avg_model8, if_global_mapping=True)[1]
        mae9 = myclients.clients_set[clients].local_test(Fed_avg_model9, if_global_mapping=False)[1]
        mae10 = myclients.clients_set[clients].local_test(Fed_avg_model10, if_global_mapping=False)[1]

        av_mae0.append(mae0 * 1000)
        av_mae1.append(mae1 * 1000)
        av_mae2.append(mae2 * 1000)
        av_mae3.append(mae3 * 1000)
        av_mae4.append(mae4 * 1000)
        av_mae5.append(mae5 * 1000)
        av_mae6.append(mae6 * 1000)
        av_mae7.append(mae7 * 1000)
        av_mae8.append(mae8 * 1000)
        av_mae9.append(mae9 * 1000)
        av_mae10.append(mae10 * 1000)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(6,3))
    plt.scatter(av_mae0, av_mae1, marker='o', color='crimson', label='Method 1', s=12)
    plt.scatter(av_mae0, av_mae2, marker='v', color='cadetblue', label='Method 2', s=12)
    plt.scatter(av_mae0, av_mae3, marker='^', color='orange', label='Method 3', s=12)
    plt.scatter(av_mae0, av_mae4, marker='1', color='royalblue', label='Method 4', s=12)
    plt.scatter(av_mae0, av_mae5, marker='2', color='cyan', label='Method 5', s=12)
    plt.scatter(av_mae0, av_mae6, marker='+', color='pink', label='Method 6', s=12)
    plt.scatter(av_mae0, av_mae7, marker='x', color='greenyellow', label='Method 7', s=12)
    plt.scatter(av_mae0, av_mae8, marker='H', color='darkgoldenrod', label='Method 8', s=12)
    plt.scatter(av_mae0, av_mae9, marker='o', color='darkorange', label='Method 9', s=12)
    plt.scatter(av_mae0, av_mae10, marker='v', color='magenta', label='Method 10', s=12)
    plt.plot([0,220], [0,220], ls='--', lw=1.5, color='black')

    plt.grid(color='silver', linestyle='-', linewidth=0.5)

    plt.xlabel('RMSE [Wh] of the proposed framework', fontproperties='Times New Roman')
    plt.ylabel('RMSE [Wh] of comparison methods', fontproperties='Times New Roman')
    plt.margins(x=0, y=0)

    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')

    font = {'family': 'Times New Roman'}
    plt.legend(prop=font, edgecolor='black', shadow=False, fancybox=False, labelspacing=0.15)
    plt.savefig(fname='Figs/points_comp_rmse_123_.eps', format='eps', bbox_inches='tight')
    plt.show()

#plot_points_rmse_com123(client_per=0.5)

def plot_val_curve(zoomx1=51, xlim=0.005, tlim=0.013):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    val_loss1 = read_csv('fedavg_val_loss_0_0.csv', usecols=[0]).values
    val_loss2 = read_csv('fedavg_val_loss_1_0.csv', usecols=[0]).values
    val_loss3 = read_csv('fedavg_val_loss_0_1.csv', usecols=[0]).values
    val_loss4 = read_csv('fedavg_val_loss_1_100.csv', usecols=[0]).values
    val_loss5 = read_csv('fedavg_without_val_loss_1_1.csv', usecols=[0]).values
    val_loss6 = read_csv('only_fed_val_loss.csv', usecols=[0]).values

    ax.plot(val_loss6, label='Method 3', color='royalblue', lw=1.5, ls='--')
    ax.plot(val_loss1, label='Method 5', color='cyan', lw=1.5, ls='--')
    ax.plot(val_loss2, label='Method 6', color='cadetblue', lw=1.5, ls='-.')
    ax.plot(val_loss3, label='Method 7', color='hotpink', lw=1.5, ls='-.', marker='')
    ax.plot(val_loss5, label='Method 8', color='orange', lw=1.5, ls='-')
    ax.plot(val_loss4, label='Proposed', color='crimson', lw=1.5, ls='-')

    ax.set_xlabel('Communication round', fontproperties='Times New Roman')
    ax.set_ylabel('Validation loss', fontproperties='Times New Roman')
    plt.margins(x=0)

    # zoom1
    zoomx2 = zoomx1 + 24
    axins_1 = ax.inset_axes((0.2, 0.55, 0.5, 0.3))
    axins_1.plot(val_loss6, color='royalblue', lw=1.5, ls='--')
    axins_1.plot(val_loss1, color='cyan', lw=1.5, ls='--')
    axins_1.plot(val_loss2, color='cadetblue', lw=1.5, ls='-.')
    axins_1.plot(val_loss3, color='hotpink', lw=1.5, ls='-.', marker='')
    axins_1.plot(val_loss5, color='orange', lw=1.5, ls='-')
    axins_1.plot(val_loss4, color='crimson', lw=1.5, ls='-')
    axins_1.set_xlim(zoomx1, zoomx2)
    axins_1.set_ylim(xlim, tlim)
    axins_1.set_title('Zoom in', fontproperties='Times New Roman')
    tx0 = zoomx1
    tx1 = zoomx2
    ty0 = xlim
    ty1 = tlim
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty0)
    xy2 = (tx0, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)
    xy = (tx1, ty0)
    xy2 = (tx1, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)

    plt.setp(axins_1.get_xticklabels(), fontproperties='Times New Roman')
    plt.setp(axins_1.get_yticklabels(), fontproperties='Times New Roman')
    plt.setp(ax.get_xticklabels(), fontproperties='Times New Roman')
    plt.setp(ax.get_yticklabels(), fontproperties='Times New Roman')

    font = {'family': 'Times New Roman'}
    plt.legend(prop=font, edgecolor='black', shadow=False, fancybox=False, labelspacing=0.15)
    plt.savefig(fname='Figs/val_loss_curve.eps', format='eps', bbox_inches='tight')
    plt.show()

#plot_val_curve()

def plot_computation_cost():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(6, 3))
    computation_time = [105.91, 4.61, 11.892, 13, 42.19,
                        80.73, 69.37, 146.66, 16.22, 156.65]
    methods = ['Proposed', 'Method 2', 'Method 3', 'Method 4',
               'Method 5', 'Method 6', 'Method 7', 'Method 8', 'Method 9', 'Method 10']
    plt.bar(range(len(computation_time)), computation_time,
            color=['green', 'teal', 'darkcyan', 'c', 'powderblue', 'lightskyblue', 'lightsteelblue',
                   'palegreen', 'lightgreen', 'limegreen'])

    plt.grid(color='silver', linestyle='-', linewidth=0.5)
    plt.ylabel('Computation cost [s]', fontproperties='Times New Roman')
    plt.margins(y=0)
    plt.ylim(0, 170)

    plt.xticks(range(len(computation_time)), labels=methods, fontproperties='Times New Roman',
               rotation=45)
    plt.yticks(fontproperties='Times New Roman')

    plt.savefig(fname='Figs/computation_cost_.eps', format='eps', bbox_inches='tight')
    plt.show()

#plot_computation_cost()

def plot_communication_cost():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(6, 3))
    computation_time = [0.05037/4*2*103*10*2, 0.010880*50*10*2, 0.010880*50*10*2, 0.05037/4*2*85*10*2, 0.05037/4*2*87*10*2,
                        0.05037/4*2*72*10*2, 0.05037/4*2*165*10*2]
    methods = ['Proposed', 'Method 3', 'Method 4', 'Method 5',
               'Method 6', 'Method 7', 'Method 8']
    plt.bar(range(len(computation_time)), computation_time,
            color=['gold', 'orange', 'tan', 'sandybrown', 'coral', 'crimson', 'pink', 'lightcoral'])

    plt.grid(color='silver', linestyle='-', linewidth=0.5)
    plt.ylabel('Communication cost [Mbyte]', fontproperties='Times New Roman')
    plt.margins(y=0)
    plt.ylim(0, 0.05037/4*2*170*10*2)

    plt.xticks(range(len(computation_time)), labels=methods, fontproperties='Times New Roman',
               rotation=45)
    plt.yticks(fontproperties='Times New Roman')

    plt.savefig(fname='Figs/communictation_cost_.eps', format='eps', bbox_inches='tight')
    plt.show()

#plot_communication_cost()

def read_mae_com123(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model0 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model5 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model6 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model7 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model8 = Fed_avg_without_personalization(input_size, feature_size, embedding_size=20,
                                                     hidden_size=args['hidden_size'],
                                                     num_layers=args['num_layers'],
                                                     num_labels=args['num_layers'],
                                                     window_width=args['window_width']).to(dev)

    num_client=1
    av_mae0, av_mae1, av_mae2, av_mae3, av_mae4, av_mae5, av_mae6, av_mae7, av_mae8 = [], [], [], [], [], [], [], [], []
    a_mae0, a_mae1, a_mae2, a_mae3, a_mae4, a_mae5, a_mae6, a_mae7, a_mae8, a_mae9, a_mae10 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        Fed_avg_model0.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 100)))
        Fed_avg_model2 = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                      format(clients)).to(dev)
        Fed_avg_model4 = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                    format(clients, client_per)).to(dev)
        Fed_avg_model3 = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                    format(client_per)).to(dev)
        Fed_avg_model5.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 0)))
        Fed_avg_model6.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 1)))
        Fed_avg_model7.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 0)))
        Fed_avg_model8.load_state_dict(
            torch.load('model_parameters_storage/fed_avg/no_personalization_model_{}_{}_{}_{}'.
                       format(clients, client_per, 1, 1)))
        Fed_avg_model9 = torch.load('model_parameters_storage/nbeats/nbeats_model_{}'.
                                    format(clients)).to(dev)
        Fed_avg_model10 = torch.load('model_parameters_storage/impactnet/impactnet_model_{}'.
                                    format(clients)).to(dev)

        mae0 = myclients.clients_set[clients].local_test(Fed_avg_model0, if_global_mapping=True)[0]
        mae1 = myclients.clients_set[clients].local_persist()[0]
        mae2 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=False)[0]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[0]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=False)[0]
        mae5 = myclients.clients_set[clients].local_test(Fed_avg_model5, if_global_mapping=True)[0]
        mae6 = myclients.clients_set[clients].local_test(Fed_avg_model6, if_global_mapping=True)[0]
        mae7 = myclients.clients_set[clients].local_test(Fed_avg_model7, if_global_mapping=True)[0]
        mae8 = myclients.clients_set[clients].local_test(Fed_avg_model8, if_global_mapping=True)[0]
        mae9 = myclients.clients_set[clients].local_test(Fed_avg_model9, if_global_mapping=False)[0]
        mae10 = myclients.clients_set[clients].local_test(Fed_avg_model10, if_global_mapping=False)[0]

        a_mae0 += mae0
        a_mae1 += mae1
        a_mae2 += mae2
        a_mae3 += mae3
        a_mae4 += mae4
        a_mae5 += mae5
        a_mae6 += mae6
        a_mae7 += mae7
        a_mae8 += mae8
        a_mae9 += mae9
        a_mae10 += mae10

        print('\#' + str(num_client), end=' &')
        print('{' + '%.2f' % (mae0 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae1 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae2 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae5 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae6 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae7 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae8 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae9 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae10 * 1000) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    a_mae0 /= myclients.num_of_clients
    a_mae1 /= myclients.num_of_clients
    a_mae2 /= myclients.num_of_clients
    a_mae3 /= myclients.num_of_clients
    a_mae4 /= myclients.num_of_clients
    a_mae5 /= myclients.num_of_clients
    a_mae6 /= myclients.num_of_clients
    a_mae7 /= myclients.num_of_clients
    a_mae8 /= myclients.num_of_clients
    a_mae9 /= myclients.num_of_clients
    a_mae10 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (a_mae0 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae1 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae2 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae3 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae4 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae5 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae6 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae7 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae8 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae9 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae10 * 1000) + '}', end=' \\' + '\\')
    print(' ')

#read_mae_com123()

def read_rmse_com123(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model0 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model5 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model6 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model7 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model8 = Fed_avg_without_personalization(input_size, feature_size, embedding_size=20,
                                                     hidden_size=args['hidden_size'],
                                                     num_layers=args['num_layers'],
                                                     num_labels=args['num_layers'],
                                                     window_width=args['window_width']).to(dev)

    num_client = 1
    av_mae0, av_mae1, av_mae2, av_mae3, av_mae4, av_mae5, av_mae6, av_mae7, av_mae8 = [], [], [], [], [], [], [], [], []
    a_mae0, a_mae1, a_mae2, a_mae3, a_mae4, a_mae5, a_mae6, a_mae7, a_mae8, a_mae9, a_mae10 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        Fed_avg_model0.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 100)))
        Fed_avg_model2 = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                    format(clients)).to(dev)
        Fed_avg_model4 = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                    format(clients, client_per)).to(dev)
        Fed_avg_model3 = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                    format(client_per)).to(dev)
        Fed_avg_model5.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 0)))
        Fed_avg_model6.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 1)))
        Fed_avg_model7.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 0)))
        Fed_avg_model8.load_state_dict(
            torch.load('model_parameters_storage/fed_avg/no_personalization_model_{}_{}_{}_{}'.
                       format(clients, client_per, 1, 1)))
        Fed_avg_model9 = torch.load('model_parameters_storage/nbeats/nbeats_model_{}'.
                                    format(clients)).to(dev)
        Fed_avg_model10 = torch.load('model_parameters_storage/impactnet/impactnet_model_{}'.
                                     format(clients)).to(dev)

        mae0 = myclients.clients_set[clients].local_test(Fed_avg_model0, if_global_mapping=True)[1]
        mae1 = myclients.clients_set[clients].local_persist()[1]
        mae2 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=False)[1]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[1]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=False)[1]
        mae5 = myclients.clients_set[clients].local_test(Fed_avg_model5, if_global_mapping=True)[1]
        mae6 = myclients.clients_set[clients].local_test(Fed_avg_model6, if_global_mapping=True)[1]
        mae7 = myclients.clients_set[clients].local_test(Fed_avg_model7, if_global_mapping=True)[1]
        mae8 = myclients.clients_set[clients].local_test(Fed_avg_model8, if_global_mapping=True)[1]
        mae9 = myclients.clients_set[clients].local_test(Fed_avg_model9, if_global_mapping=False)[1]
        mae10 = myclients.clients_set[clients].local_test(Fed_avg_model10, if_global_mapping=False)[1]

        a_mae0 += mae0
        a_mae1 += mae1
        a_mae2 += mae2
        a_mae3 += mae3
        a_mae4 += mae4
        a_mae5 += mae5
        a_mae6 += mae6
        a_mae7 += mae7
        a_mae8 += mae8
        a_mae9 += mae9
        a_mae10 += mae10

        print('\#' + str(num_client), end=' &')
        print('{' + '%.2f' % (mae0 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae1 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae2 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae3 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae4 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae5 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae6 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae7 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae8 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae9 * 1000) + '}', end=' &')
        print('{' + '%.2f' % (mae10 * 1000) + '}', end=' \\' + '\\')
        print(' ')
        num_client += 1

    a_mae0 /= myclients.num_of_clients
    a_mae1 /= myclients.num_of_clients
    a_mae2 /= myclients.num_of_clients
    a_mae3 /= myclients.num_of_clients
    a_mae4 /= myclients.num_of_clients
    a_mae5 /= myclients.num_of_clients
    a_mae6 /= myclients.num_of_clients
    a_mae7 /= myclients.num_of_clients
    a_mae8 /= myclients.num_of_clients
    a_mae9 /= myclients.num_of_clients
    a_mae10 /= myclients.num_of_clients

    print('\midrule')
    print('Average', end=' &')
    print('{' + '%.2f' % (a_mae0 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae1 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae2 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae3 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae4 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae5 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae6 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae7 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae8 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae9 * 1000) + '}', end=' &')
    print('{' + '%.2f' % (a_mae10 * 1000) + '}', end=' \\' + '\\')
    print(' ')

#read_rmse_com123()

def plot_MMD_DIFF_curve(zoomx1=71, xlim=-0.001, tlim=0.006):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    val_loss4 = read_csv('fedavg_val_mmd_1_100.csv', usecols=[0]).values[1:]
    val_loss5 = read_csv('fedavg_val_diff_1_100.csv', usecols=[0]).values[0:-1]*10

    ax.plot(val_loss4, label='MMD constraint loss', color='cadetblue', lw=1.5, ls='-')
    ax.plot(val_loss5, label='Orthogonal constraint loss', color='crimson', lw=1.5, ls='-')

    ax.set_xlabel('Communication round', fontproperties='Times New Roman')
    ax.set_ylabel('Validation loss', fontproperties='Times New Roman')
    plt.margins(x=0)
    plt.ylim(-0.003, 0.035)

    # zoom1
    zoomx2 = zoomx1 + 24
    axins_1 = ax.inset_axes((0.55, 0.4, 0.4, 0.25))
    axins_1.plot(val_loss4, color='cadetblue', lw=1.5, ls='-')
    axins_1.plot(val_loss5, color='crimson', lw=1.5, ls='-')
    axins_1.set_xlim(zoomx1, zoomx2)
    axins_1.set_ylim(xlim, tlim)
    axins_1.set_title('Zoom in', fontproperties='Times New Roman')
    tx0 = zoomx1
    tx1 = zoomx2
    ty0 = xlim
    ty1 = tlim
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty0)
    xy2 = (tx0, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)
    xy = (tx1, ty0)
    xy2 = (tx1, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)

    plt.setp(axins_1.get_xticklabels(), fontproperties='Times New Roman')
    plt.setp(axins_1.get_yticklabels(), fontproperties='Times New Roman')
    plt.setp(ax.get_xticklabels(), fontproperties='Times New Roman')
    plt.setp(ax.get_yticklabels(), fontproperties='Times New Roman')

    font = {'family': 'Times New Roman'}
    plt.legend(prop=font, edgecolor='black', shadow=False, fancybox=False, labelspacing=0.15)
    plt.savefig(fname='Figs/val_mmd_diff_curve.eps', format='eps', bbox_inches='tight')
    plt.show()

#plot_MMD_DIFF_curve()

def plot_points_com123_png(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model0 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model5 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model6 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model7 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model8 = Fed_avg_without_personalization(input_size, feature_size, embedding_size=20,
                                                     hidden_size=args['hidden_size'],
                                                     num_layers=args['num_layers'],
                                                     num_labels=args['num_layers'],
                                                     window_width=args['window_width']).to(dev)

    num_client=1
    av_mae0, av_mae1, av_mae2, av_mae3, av_mae4, av_mae5, av_mae6, av_mae7, av_mae8, av_mae9, av_mae10  = [], [], [], [], [], [], [], [], [], [], []
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        Fed_avg_model0.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 100)))
        Fed_avg_model2 = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                      format(clients)).to(dev)
        Fed_avg_model3 = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                    format(clients, client_per)).to(dev)
        Fed_avg_model4 = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                    format(client_per)).to(dev)
        Fed_avg_model5.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 0)))
        Fed_avg_model6.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 1)))
        Fed_avg_model7.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 0)))
        Fed_avg_model8.load_state_dict(
            torch.load('model_parameters_storage/fed_avg/no_personalization_model_{}_{}_{}_{}'.
                       format(clients, client_per, 1, 1)))
        Fed_avg_model9 = torch.load('model_parameters_storage/nbeats/nbeats_model_{}'.
                                    format(clients)).to(dev)
        Fed_avg_model10 = torch.load('model_parameters_storage/impactnet/impactnet_model_{}'.
                                    format(clients)).to(dev)

        mae0 = myclients.clients_set[clients].local_test(Fed_avg_model0, if_global_mapping=True)[0]
        mae1 = myclients.clients_set[clients].local_persist()[0]
        mae2 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=False)[0]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[0]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=False)[0]
        mae5 = myclients.clients_set[clients].local_test(Fed_avg_model5, if_global_mapping=True)[0]
        mae6 = myclients.clients_set[clients].local_test(Fed_avg_model6, if_global_mapping=True)[0]
        mae7 = myclients.clients_set[clients].local_test(Fed_avg_model7, if_global_mapping=True)[0]
        mae8 = myclients.clients_set[clients].local_test(Fed_avg_model8, if_global_mapping=True)[0]
        mae9 = myclients.clients_set[clients].local_test(Fed_avg_model9, if_global_mapping=False)[0]
        mae10 = myclients.clients_set[clients].local_test(Fed_avg_model10, if_global_mapping=False)[0]

        av_mae0.append(mae0 * 1000)
        av_mae1.append(mae1*1000)
        av_mae2.append(mae2 * 1000)
        av_mae3.append(mae3*1000)
        av_mae4.append(mae4*1000)
        av_mae5.append(mae5 * 1000)
        av_mae6.append(mae6 * 1000)
        av_mae7.append(mae7 * 1000)
        av_mae8.append(mae8 * 1000)
        av_mae9.append(mae9 * 1000)
        av_mae10.append(mae10 * 1000)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(6,3))
    plt.scatter(av_mae0, av_mae1, marker='o', color='green', label='Method 1', s=12)
    plt.scatter(av_mae0, av_mae2, marker='v', color='cadetblue', label='Method 2', s=12)
    plt.scatter(av_mae0, av_mae3, marker='^', color='blueviolet', label='Method 3', s=12)
    plt.scatter(av_mae0, av_mae4, marker='1', color='royalblue', label='Method 4', s=12)
    plt.scatter(av_mae0, av_mae5, marker='2', color='cyan', label='Method 5', s=12)
    #plt.scatter(av_mae0, av_mae6, marker='+', color='pink', label='Method 6', s=12)
    #plt.scatter(av_mae0, av_mae7, marker='x', color='greenyellow', label='Method 7', s=12)
    #plt.scatter(av_mae0, av_mae8, marker='H', color='darkgoldenrod', label='Method 8', s=12)
    plt.scatter(av_mae0, av_mae9, marker='o', color='darkorange', label='Method 6', s=12)
    plt.scatter(av_mae0, av_mae10, marker='v', color='magenta', label='Method 7', s=12)
    plt.plot([0,175], [0,175], ls='--', lw=1.5, color='black')

    plt.grid(color='silver', linestyle='-', linewidth=0.5)

    plt.xlabel('MAE [Wh] of the proposed framework', fontproperties='Times New Roman')
    plt.ylabel('MAE [Wh] of comparison methods', fontproperties='Times New Roman')
    plt.margins(x=0, y=0)

    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')

    font = {'family': 'Times New Roman'}
    plt.legend(prop=font, edgecolor='black', shadow=False, fancybox=False, labelspacing=0.15)
    plt.savefig(fname='Figs/points_comp_123_.png', format='png', bbox_inches='tight')
    plt.show()

plot_points_com123_png(client_per=0.5)

def plot_points_rmse_com123_png(dev=myclients.dev, client_per=0.5):
    print('client_per:', client_per)
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    Fed_avg_model0 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model5 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model6 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model7 = Fed_avg(input_size, feature_size, embedding_size=10, hidden_size=args['hidden_size'],
                             num_layers=args['num_layers'],
                             num_labels=args['num_layers'], window_width=args['window_width']).to(dev)
    Fed_avg_model8 = Fed_avg_without_personalization(input_size, feature_size, embedding_size=20,
                                                     hidden_size=args['hidden_size'],
                                                     num_layers=args['num_layers'],
                                                     num_labels=args['num_layers'],
                                                     window_width=args['window_width']).to(dev)

    num_client = 1
    av_mae0, av_mae1, av_mae2, av_mae3, av_mae4, av_mae5, av_mae6, av_mae7, av_mae8, av_mae9, av_mae10 = [], [], [], [], [], [], [], [], [], [], []
    for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
        Fed_avg_model0.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 100)))
        Fed_avg_model2 = torch.load('model_parameters_storage/individual/individual_model_{}'.
                                    format(clients)).to(dev)
        Fed_avg_model3 = torch.load('model_parameters_storage/local_fine_tune/individual_model_{}_{}'.
                                    format(clients, client_per)).to(dev)
        Fed_avg_model4 = torch.load('model_parameters_storage/only_fed/global_model_{}'.
                                    format(client_per)).to(dev)
        Fed_avg_model5.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 0)))
        Fed_avg_model6.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 0, 1)))
        Fed_avg_model7.load_state_dict(torch.load('model_parameters_storage/fed_avg/individual_model_{}_{}_{}_{}'.
                                                  format(clients, client_per, 1, 0)))
        Fed_avg_model8.load_state_dict(
            torch.load('model_parameters_storage/fed_avg/no_personalization_model_{}_{}_{}_{}'.
                       format(clients, client_per, 1, 1)))
        Fed_avg_model9 = torch.load('model_parameters_storage/nbeats/nbeats_model_{}'.
                                    format(clients)).to(dev)
        Fed_avg_model10 = torch.load('model_parameters_storage/impactnet/impactnet_model_{}'.
                                     format(clients)).to(dev)

        mae0 = myclients.clients_set[clients].local_test(Fed_avg_model0, if_global_mapping=True)[1]
        mae1 = myclients.clients_set[clients].local_persist()[1]
        mae2 = myclients.clients_set[clients].local_test(Fed_avg_model2, if_global_mapping=False)[1]
        mae3 = myclients.clients_set[clients].local_test(Fed_avg_model3, if_global_mapping=False)[1]
        mae4 = myclients.clients_set[clients].local_test(Fed_avg_model4, if_global_mapping=False)[1]
        mae5 = myclients.clients_set[clients].local_test(Fed_avg_model5, if_global_mapping=True)[1]
        mae6 = myclients.clients_set[clients].local_test(Fed_avg_model6, if_global_mapping=True)[1]
        mae7 = myclients.clients_set[clients].local_test(Fed_avg_model7, if_global_mapping=True)[1]
        mae8 = myclients.clients_set[clients].local_test(Fed_avg_model8, if_global_mapping=True)[1]
        mae9 = myclients.clients_set[clients].local_test(Fed_avg_model9, if_global_mapping=False)[1]
        mae10 = myclients.clients_set[clients].local_test(Fed_avg_model10, if_global_mapping=False)[1]

        av_mae0.append(mae0 * 1000)
        av_mae1.append(mae1 * 1000)
        av_mae2.append(mae2 * 1000)
        av_mae3.append(mae3 * 1000)
        av_mae4.append(mae4 * 1000)
        av_mae5.append(mae5 * 1000)
        av_mae6.append(mae6 * 1000)
        av_mae7.append(mae7 * 1000)
        av_mae8.append(mae8 * 1000)
        av_mae9.append(mae9 * 1000)
        av_mae10.append(mae10 * 1000)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(6,3))
    plt.scatter(av_mae0, av_mae1, marker='o', color='green', label='Method 1', s=12)
    plt.scatter(av_mae0, av_mae2, marker='v', color='cadetblue', label='Method 2', s=12)
    plt.scatter(av_mae0, av_mae3, marker='^', color='blueviolet', label='Method 3', s=12)
    plt.scatter(av_mae0, av_mae4, marker='1', color='royalblue', label='Method 4', s=12)
    plt.scatter(av_mae0, av_mae5, marker='2', color='cyan', label='Method 5', s=12)
    #plt.scatter(av_mae0, av_mae6, marker='+', color='pink', label='Method 6', s=12)
    #plt.scatter(av_mae0, av_mae7, marker='x', color='greenyellow', label='Method 7', s=12)
    #plt.scatter(av_mae0, av_mae8, marker='H', color='darkgoldenrod', label='Method 8', s=12)
    plt.scatter(av_mae0, av_mae9, marker='o', color='darkorange', label='Method 6', s=12)
    plt.scatter(av_mae0, av_mae10, marker='v', color='magenta', label='Method 7', s=12)
    plt.plot([0,220], [0,220], ls='--', lw=1.5, color='black')

    plt.grid(color='silver', linestyle='-', linewidth=0.5)

    plt.xlabel('RMSE [Wh] of the proposed framework', fontproperties='Times New Roman')
    plt.ylabel('RMSE [Wh] of comparison methods', fontproperties='Times New Roman')
    plt.margins(x=0, y=0)

    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')

    font = {'family': 'Times New Roman'}
    plt.legend(prop=font, edgecolor='black', shadow=False, fancybox=False, labelspacing=0.15)
    plt.savefig(fname='Figs/points_comp_rmse_123_.png', format='png', bbox_inches='tight')
    plt.show()

plot_points_rmse_com123_png(client_per=0.5)