import time

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
from server import Fedavg_server, individual, only_Fed, local_fine_tune, Fedavg_without_personalization_server, nbeats, impactnet_train
from client import client_group
from read_results import test_fedavg, test_individual, test_local_fine_tune, test_only_fed, test_local_persist, test_nbeats, test_impactnet
from experimental_parameters import args


#parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
#parser.add_argument('-cn', '--client_num', type=int, default=30,
#                    help='total number of the participated clients')
#parser.add_argument('-cp', '--client_per_space', type=list, default=[0.3],
#                    help='space of the percentage of clients in every communication round')
#parser.add_argument('-as', '--alpha_searching_space', type=list, default=[0.5],
#                    help='searching space of the hyperparameter: alpha')
#parser.add_argument('-bs', '--beta_searching_space', type=list, default=[1],
#                    help='searching space of the hyperparameter: beta')

#print(torch.cuda.is_available())

if __name__=="__main__":
    #args = parser.parse_args()
    #args = args.__dict__
    start = time.time()
    myclients = client_group(args['client_num'], args['device'], window_width=args['window_width'])
    end = time.time()
    print('load time cost:', end-start)
    for client_per in args['client_per_space']:
        for alpha in args['alpha_searching_space']:
            for beta in args['beta_searching_space']:
                print('client_per: ', client_per, 'alpha: ', alpha, 'beta: ', beta)
                #Fedavg_server(client_set=myclients, client_num=args['client_num'], client_per=client_per, alpha=alpha,
                #              beta=beta, hidden_size=args['hidden_size'], embedding_size=10,
                #              num_layers=args['num_layers'], comm_rounds=500, local_batch=256,
                #              local_epoch=5, window_width=args['window_width'])
        #only_Fed(client_set=myclients, client_num=args['client_num'], client_per=client_per,
        #                hidden_size=args['hidden_size'], local_batch=256, forecast_period=1,
        #                num_layers=args['num_layers'], local_epoch=5)
        #local_fine_tune(client_set=myclients, client_num=args['client_num'], client_per=client_per,
        #                hidden_size=args['hidden_size'], local_batch=256, forecast_period=1,
        #                num_layers=args['num_layers'], local_epoch=5)
    #individual(client_set=myclients, hidden_size=args['hidden_size'], client_num=args['client_num'],
    #           window_width=args['window_width'], local_batch=256, forecast_period=1,
    #           num_layers=args['num_layers'])
    #Fedavg_without_personalization_server(client_set=myclients, hidden_size=args['hidden_size'], client_num=args['client_num'],
    #           window_width=args['window_width'], local_batch=256, forecast_period=1,
    #           num_layers=args['num_layers'], local_epoch=5)
    #nbeats(client_set=myclients, hidden_size=args['hidden_size'], client_num=args['client_num'],
    #           window_width=args['window_width'], local_batch=64, forecast_period=1,
    #           num_layers=args['num_layers'])
    #impactnet_train(client_set=myclients, hidden_size=args['hidden_size'], client_num=args['client_num'],
    #                  window_width=args['window_width'], local_batch=64, forecast_period=1,
    #                  num_layers=args['num_layers'])


    #start = time.time()
    test_fedavg(myclients=myclients, hidden_size=args['hidden_size'], embedding_size=10,
                num_layers=args['num_layers'], window_width=args['window_width'])
    #end = time.time()
    #print('infer time cost:', end-start)

    test_individual(myclients)
    #test_local_fine_tune(myclients)
    test_only_fed(myclients)
    #test_local_persist(myclients)
    #test_nbeats(myclients)
    #test_impactnet(myclients)

