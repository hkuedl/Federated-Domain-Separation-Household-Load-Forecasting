import numpy as np
import argparse
import os
import csv
import time

#"1, 2"表示训练的时候选用两块GPU，优先选用"1"号GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import torch
from torch.utils.data import DataLoader
from Forecasting_Models import mmd, diff_loss, global_mapping, local_align_mapping, local_orthogonal_mapping,\
    Fed_avg, init_global_mapping, Fed_avg_without_personalization, NBEATS, impactnet
from client import client, client_group, same_seeds
import torch.nn as nn
import matplotlib.pyplot as plt
from experimental_parameters import args

class Fed_avg_EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_parameters = None

    def __call__(self, val_loss, model, clients_set, client_per, global_parameters, alpha, beta, save_path='fed_avg'):
        """
        Args:
            val_loss (float): validation loss for a communication round
            model (nn.Module): the type of the individual model
            clients_set: clients set which contains of all client objects
            client_per (float): the percentage of clients involved in a communication round
            global_parameters: global_parameters in this round
            save_path (str): where to save the model
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, clients_set, model, save_path, client_per, global_parameters, alpha, beta)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'Communication EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, clients_set, model, save_path, client_per, global_parameters, alpha, beta)
            self.counter = 0

    def save_checkpoint(self, val_loss, myclients, model, save_path, client_per, global_parameters, alpha, beta):
        '''Saves model when validation loss decrease.'''
        #if self.verbose:
        #    print(f'Communication Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # global_parameters are distributed to all individual models, and then save them
        for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
            model.load_state_dict(myclients.clients_set[clients].local_parameters, strict=True)
            model.global_mapping.load_state_dict(global_parameters, strict=True)
            torch.save(model.state_dict(), 'model_parameters_storage/'+save_path+'/individual_model_{}_{}_{}_{}'.
                           format(clients, client_per, alpha, beta))
        self.best_parameters = model.state_dict()
        self.val_loss_min = val_loss

class benchmark_EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_parameters = None

    def __call__(self, val_loss, model):
        """
        Args:
            val_loss (float): validation loss for a communication round
            model (nn.Module): the type of the individual model
            clients_set: clients set which contains of all client objects
            client_per (float): the percentage of clients involved in a communication round
            save_path (str): where to save the model
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'Communication EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        #if self.verbose:
        #    print(f'Communication Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_parameters = model.state_dict()
        self.val_loss_min = val_loss

class Fed_avg_without_personalization_EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=25, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_parameters = None

    def __call__(self, val_loss, model, clients_set, client_per, global_parameters, alpha, beta, save_path='fed_avg'):
        """
        Args:
            val_loss (float): validation loss for a communication round
            model (nn.Module): the type of the individual model
            clients_set: clients set which contains of all client objects
            client_per (float): the percentage of clients involved in a communication round
            global_parameters: global_parameters in this round
            save_path (str): where to save the model
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, clients_set, model, save_path, client_per, global_parameters, alpha, beta)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'Communication EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, clients_set, model, save_path, client_per, global_parameters, alpha, beta)
            self.counter = 0

    def save_checkpoint(self, val_loss, myclients, model, save_path, client_per, global_parameters, alpha, beta):
        '''Saves model when validation loss decrease.'''
        #if self.verbose:
        #    print(f'Communication Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # global_parameters are distributed to all individual models, and then save them
        for clients in ['client{}'.format(i) for i in range(myclients.num_of_clients)]:
            model.load_state_dict(myclients.clients_set[clients].local_parameters, strict=True)
            model.global_mapping.load_state_dict(global_parameters, strict=True)
            torch.save(model.state_dict(), 'model_parameters_storage/'+save_path+'/no_personalization_model_{}_{}_{}_{}'.
                           format(clients, client_per, alpha, beta))
        self.best_parameters = model.state_dict()
        self.val_loss_min = val_loss

def Fedavg_server(client_set, dev=args['device'], client_num=20, client_per=0.5,
                  comm_rounds=500, embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1, window_width=12,
                  local_epoch=20, local_batch=64, alpha=0.2, beta=1):
    same_seeds()
    np.random.seed(6)


    time_start = time.time()

    f = open('fedavg_val_loss_{}_{}.csv'.format(alpha, beta), 'w', encoding='utf-8', newline="")
    writer = csv.writer(f)
    f1 = open('fedavg_val_mmd_{}_{}.csv'.format(alpha, beta), 'w', encoding='utf-8', newline="")
    writer1 = csv.writer(f1)
    f2 = open('fedavg_val_diff_{}_{}.csv'.format(alpha, beta), 'w', encoding='utf-8', newline="")
    writer2 = csv.writer(f2)
    writer.writerow(['val_loss'])
    writer1.writerow(['mmd_loss'])
    writer2.writerow(['diff_loss'])

    print('Fedavg_server start')

    # generate the clients set
    #myclients = client_group(client_num, dev)
    myclients = client_set
    # get input size
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    # build the models
    # to avoid too many models being sent to the device, only parameters are updated and saved in training phase
    global_mapping_model = global_mapping(input_size, feature_size, embedding_size, hidden_size, num_layers, window_width=window_width).to(dev)
    Fed_avg_model = Fed_avg(input_size, feature_size, embedding_size, hidden_size, num_layers, forecast_period, window_width=window_width).to(dev)
    init_global_mapping_model = init_global_mapping(input_size, feature_size, embedding_size,
                                                    hidden_size, num_layers, forecast_period, window_width=window_width).to(dev)
    init_parameters = init_global_mapping_model.state_dict()

    # set the earlystopping
    if alpha == 1 and beta ==100:
        earlystopping = Fed_avg_EarlyStopping(patience=15, delta=0)
    else:
        earlystopping = Fed_avg_EarlyStopping(patience=15, delta=0.0001)

    # % the parameters of global mapping will not be updated during client training
    for p in Fed_avg_model.global_mapping.parameters():
        p.requires_grad = False

    # record global_mapping parameters
    global_mapping_parameters = {}
    for key, var in global_mapping_model.state_dict().items():
        global_mapping_parameters[key] = var.clone()

    # start the communication
    for cumm_round in range(comm_rounds):

        print('comm_round: ', cumm_round)
        sum_parameters = None
        # initialize the global_mapping_model with all clients
        # training -> save global_mapping_model -> sum parameters -> calculate global_mapping_parameters
        if cumm_round == 0:
            clients_in_comm = ['client{}'.format(i) for i in range(client_num)]
            for clients in clients_in_comm:
                init_global_mapping_model.load_state_dict(init_parameters, strict=True)
                myclients.clients_set[clients].localInitialize(10, local_batch, init_global_mapping_model,
                                                               torch.optim.Adam(init_global_mapping_model
                                                                                .parameters(), lr=0.0001))
                # initialize local_parameters for each client
                myclients.clients_set[clients].local_parameters = Fed_avg_model.state_dict()
                # in the first round, calculate the summation of global_mapping parameters in init_global_mapping_models
                client_mapping_parameters = init_global_mapping_model.global_mapping.state_dict()
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in client_mapping_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += client_mapping_parameters[var]

            for var in global_mapping_parameters:
                global_mapping_parameters[var] = (sum_parameters[var] / client_num)

        # update Fed_avg_model with a specific proportion of clients
        # load global_mapping_parameters and client's private parameters -> training -> save Fed_avg_model
        # -> calculate global_mapping_parameters
        else:
            order = np.random.permutation(client_num)
            # select clients to be trained
            clients_in_comm = ['client{}'.format(i) for i in order[0:int(client_num*client_per)]]
            for clients in clients_in_comm:
                # in the second round, load local_parameters and global_mapping_parameters to each client
                Fed_avg_model.load_state_dict(myclients.clients_set[clients].local_parameters, strict=True)
                Fed_avg_model.global_mapping.load_state_dict(global_mapping_parameters, strict=True)
                myclients.clients_set[clients].localUpdate(local_epoch, local_batch, Fed_avg_model,
                                                           torch.optim.Adam(Fed_avg_model.parameters(), lr=0.0001),
                                                           alpha, beta)

            # collect all the local_align_mapping parameters to calculate global_mapping parameters
            val_loss = 0
            MMD = 0
            DIFF = 0
            for clients in ['client{}'.format(i) for i in range(client_num)]:
                Fed_avg_model.load_state_dict(myclients.clients_set[clients].local_parameters, strict=True)
                val_loss += myclients.clients_set[clients].local_val(Fed_avg_model, alpha, beta, if_local_mapping=True)[0]
                MMD += myclients.clients_set[clients].local_val(Fed_avg_model, alpha, beta, if_local_mapping=True)[
                    1]
                DIFF += myclients.clients_set[clients].local_val(Fed_avg_model, alpha, beta, if_local_mapping=True)[
                    2]
                client_mapping_parameters = Fed_avg_model.local_align_mapping.state_dict()
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in client_mapping_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += client_mapping_parameters[var]

            for var in global_mapping_parameters:
                global_mapping_parameters[var] = (sum_parameters[var] / client_num)

            val_loss /= client_num
            print(MMD)
            print(DIFF)

            # distribute global_parameters to all clients to uniform individual models
            # if val_loss does not decrease then early stopping and save the best individual models
            earlystopping(val_loss, Fed_avg_model, myclients, client_per, global_mapping_parameters, alpha, beta)
            writer.writerow([float(val_loss)])
            writer1.writerow([float(MMD)])
            writer2.writerow([float(DIFF)])
            print(val_loss)
            if earlystopping.early_stop:
                time_end = time.time()
                print('time cost: ', time_end-time_start)
                print('communication Early stopping')
                break

def individual(client_set, dev=args['device'], client_num=20,
                     embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1, window_width=24,
                    local_epoch=40, local_batch=64):
    same_seeds()
    np.random.seed(6)

    # generate all the clients set
    #myclients = client_group(client_num, dev)
    myclients = client_set

    time_start = time.time()

    # get input size
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    # build the models
    init_global_mapping_model = init_global_mapping(input_size, feature_size, embedding_size,
                                                    hidden_size, num_layers, forecast_period, window_width=window_width).to(dev)
    init_parameters = init_global_mapping_model.state_dict()

    clients_in_comm = ['client{}'.format(i) for i in range(client_num)]
    for clients in clients_in_comm:
        init_global_mapping_model.load_state_dict(init_parameters, strict=True)
        myclients.clients_set[clients].localInitialize(local_epoch, local_batch, init_global_mapping_model,
                                                       torch.optim.Adam(init_global_mapping_model
                                                                        .parameters(), lr=0.001), pa=7)
        torch.save(init_global_mapping_model, 'model_parameters_storage/individual/individual_model_{}'
                   .format(clients))
    time_end = time.time()
    print('time cost: ', time_end-time_start)

def only_Fed(client_set, dev=args['device'], client_num=20, client_per=0.5,
                  comm_rounds=500, embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1, window_width=24,
                  local_epoch=5, local_batch=64,):
    same_seeds()
    np.random.seed(6)

    time_start = time.time()

    f = open('only_fed_val_loss.csv', 'w', encoding='utf-8', newline="")
    writer = csv.writer(f)
    writer.writerow(['val_loss'])

    # generate all the clients set
    #myclients = client_group(client_num, dev)
    myclients = client_set

    # get input size
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    # build the models
    init_global_mapping_model = init_global_mapping(input_size, feature_size, embedding_size,
                                                    hidden_size, num_layers, forecast_period, window_width=window_width).to(dev)
    init_parameters = init_global_mapping_model.state_dict()

    # set earlystopping
    earlystopping = benchmark_EarlyStopping(patience=10)

    # record global_mapping parameters
    global_mapping_parameters = {}
    for key, var in init_global_mapping_model.state_dict().items():
        global_mapping_parameters[key] = var.clone()

    # start the communication
    for cumm_round in range(comm_rounds):

        print(cumm_round)
        sum_parameters = None
        # initialize the global_mapping_model with all clients
        # training -> save global_mapping_model -> sum parameters -> calculate global_mapping_parameters
        if cumm_round == 0:
            clients_in_comm = ['client{}'.format(i) for i in range(client_num)]
            for clients in clients_in_comm:
                init_global_mapping_model.load_state_dict(init_parameters, strict=True)
                myclients.clients_set[clients].localInitialize(local_epoch, local_batch, init_global_mapping_model,
                                                               torch.optim.Adam(init_global_mapping_model
                                                                                .parameters(), lr=0.0001))
                client_parameters = init_global_mapping_model.state_dict()
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in client_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += client_parameters[var]

            for var in global_mapping_parameters:
                global_mapping_parameters[var] = (sum_parameters[var] / client_num)

        # update Fed_avg_model with a specific proportion of clients
        # load global_mapping_parameters and client's private parameters -> training -> save Fed_avg_model
        # -> calculate global_mapping_parameters
        else:
            order = np.random.permutation(client_num)
            clients_in_comm = ['client{}'.format(i) for i in order[0:int(client_num*client_per)]]
            for clients in clients_in_comm:
                init_global_mapping_model.load_state_dict(global_mapping_parameters, strict=True)
                myclients.clients_set[clients].localInitialize(10, local_batch, init_global_mapping_model,
                                                               torch.optim.Adam(init_global_mapping_model
                                                                                .parameters(), lr=0.001))
                myclients.clients_set[clients].local_val(init_global_mapping_model, if_local_mapping=False)
                client_mapping_parameters = init_global_mapping_model.state_dict()
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in client_mapping_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += client_mapping_parameters[var]

            for var in global_mapping_parameters:
                global_mapping_parameters[var] = (sum_parameters[var] / len(clients_in_comm))

            # validation
            val_loss = 0
            for clients in ['client{}'.format(i) for i in range(client_num)]:
                init_global_mapping_model.load_state_dict(global_mapping_parameters, strict=True)
                val_loss += myclients.clients_set[clients].local_val(init_global_mapping_model, if_local_mapping=False)
            val_loss /= client_num
            writer.writerow([float(val_loss)])
            earlystopping(val_loss, init_global_mapping_model)
            if earlystopping.early_stop:
                time_end = time.time()
                print('time cost: ', time_end-time_start)
                break

    init_global_mapping_model.load_state_dict(earlystopping.best_parameters, strict=True)
    torch.save(init_global_mapping_model, 'model_parameters_storage/only_fed/global_model_{}'.format(client_per))

def local_fine_tune(client_set, dev=args['device'], client_num=20, client_per=0.5,
                  comm_rounds=500, embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1, window_width=24,
                  local_epoch=5, local_batch=64):
    same_seeds()
    np.random.seed(6)

    # generate all the clients set
    myclients = client_set

    time_start = time.time()

    # get input size
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    # build the models
    init_global_mapping_model = init_global_mapping(input_size, feature_size, embedding_size,
                                                    hidden_size, num_layers, forecast_period, window_width=window_width).to(dev)
    init_parameters = init_global_mapping_model.state_dict()

    # set early stopping
    earlystopping = benchmark_EarlyStopping()

    # record global_mapping parameters
    global_mapping_parameters = {}
    for key, var in init_global_mapping_model.state_dict().items():
        global_mapping_parameters[key] = var.clone()

    # start the communication
    for cumm_round in range(comm_rounds):

        print(cumm_round)
        sum_parameters = None
        # initialize the global_mapping_model with all clients
        # training -> save global_mapping_model -> sum parameters -> calculate global_mapping_parameters
        if cumm_round == 0:
            clients_in_comm = ['client{}'.format(i) for i in range(client_num)]
            for clients in clients_in_comm:
                init_global_mapping_model.load_state_dict(init_parameters, strict=True)
                myclients.clients_set[clients].localInitialize(local_epoch, local_batch, init_global_mapping_model,
                                                               torch.optim.Adam(init_global_mapping_model
                                                                                .parameters(), lr=0.001))
                client_mapping_parameters = init_global_mapping_model.state_dict()
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in client_mapping_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += client_mapping_parameters[var]

                for var in global_mapping_parameters:
                    global_mapping_parameters[var] = (sum_parameters[var] / client_num)

        # update Fed_avg_model with a specific proportion of clients
        # load global_mapping_parameters and client's private parameters -> training -> save Fed_avg_model
        # -> calculate global_mapping_parameters
        else:
            order = np.random.permutation(client_num)
            #print(order)
            clients_in_comm = ['client{}'.format(i) for i in order[0:int(client_num*client_per)]]
            for clients in clients_in_comm:
                init_global_mapping_model.load_state_dict(global_mapping_parameters, strict=True)
                myclients.clients_set[clients].localInitialize(local_epoch, local_batch, init_global_mapping_model,
                                                               torch.optim.Adam(init_global_mapping_model
                                                                                .parameters(), lr=0.001))
                #torch.save(init_global_mapping_model, 'model_parameters_storage/local_fine_tune/individual_model_{}_{}_'.
                #           format(clients, client_per))
                client_mapping_parameters = init_global_mapping_model.state_dict()
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in client_mapping_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += client_mapping_parameters[var]

            for var in global_mapping_parameters:
                global_mapping_parameters[var] = (sum_parameters[var] / len(clients_in_comm))

            # validation
            val_loss = 0
            for clients in ['client{}'.format(i) for i in range(client_num)]:
                init_global_mapping_model.load_state_dict(global_mapping_parameters, strict=True)
                val_loss += myclients.clients_set[clients].local_val(init_global_mapping_model, if_local_mapping=False)
            val_loss /= client_num
            print(val_loss)
            earlystopping(val_loss, init_global_mapping_model)
            if earlystopping.early_stop:
                break

    # After communication, locally fine-tune the model parameters
    clients_in_comm = ['client{}'.format(i) for i in range(client_num)]
    for clients in clients_in_comm:
        init_global_mapping_model.load_state_dict(earlystopping.best_parameters, strict=True)
        myclients.clients_set[clients].localInitialize(10, local_batch, init_global_mapping_model,
                                                       torch.optim.Adam(init_global_mapping_model
                                                                        .parameters(), lr=0.001), pa=15)

        torch.save(init_global_mapping_model, 'model_parameters_storage/local_fine_tune/individual_model_{}_{}'
                   .format(clients, client_per))

    time_end = time.time()
    print('time cost: ', time_end-time_start)

def Fedavg_without_personalization_server(client_set, dev=args['device'], client_num=20, client_per=0.5,
                  comm_rounds=500, embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1, window_width=12,
                  local_epoch=5, local_batch=64, alpha=1, beta=1):
    same_seeds()
    np.random.seed(6)

    print('Fedavg_server start')

    time_start = time.time()

    f = open('fedavg_without_val_loss_{}_{}.csv'.format(alpha, beta), 'w', encoding='utf-8', newline="")
    writer = csv.writer(f)
    writer.writerow(['val_loss'])

    # generate the clients set
    #myclients = client_group(client_num, dev)
    myclients = client_set
    # get input size
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    # build the models
    # to avoid too many models being sent to the device, only parameters are updated and saved in training phase
    global_mapping_model = global_mapping(input_size, feature_size, embedding_size, hidden_size, num_layers, window_width=window_width).to(dev)
    Fed_avg_model = Fed_avg_without_personalization(input_size, feature_size, embedding_size, hidden_size, num_layers, forecast_period, window_width=window_width).to(dev)
    init_global_mapping_model = init_global_mapping(input_size, feature_size, embedding_size,
                                                    hidden_size, num_layers, forecast_period, window_width=window_width).to(dev)
    init_parameters = init_global_mapping_model.state_dict()

    # set the earlystopping
    if alpha == 1 and beta ==1:
        earlystopping = Fed_avg_without_personalization_EarlyStopping(patience=25)
    else:
        earlystopping = Fed_avg_without_personalization_EarlyStopping(patience=15, delta=0.0001)

    # % the parameters of global mapping will not be updated during client training
    for p in Fed_avg_model.global_mapping.parameters():
        p.requires_grad = False

    # record global_mapping parameters
    global_mapping_parameters = {}
    for key, var in global_mapping_model.state_dict().items():
        global_mapping_parameters[key] = var.clone()

    # start the communication
    for cumm_round in range(comm_rounds):

        print('comm_round: ', cumm_round)
        sum_parameters = None
        # initialize the global_mapping_model with all clients
        # training -> save global_mapping_model -> sum parameters -> calculate global_mapping_parameters
        if cumm_round == 0:
            clients_in_comm = ['client{}'.format(i) for i in range(client_num)]
            for clients in clients_in_comm:
                init_global_mapping_model.load_state_dict(init_parameters, strict=True)
                myclients.clients_set[clients].localInitialize(10, local_batch, init_global_mapping_model,
                                                               torch.optim.Adam(init_global_mapping_model
                                                                                .parameters(), lr=0.0001))
                # initialize local_parameters for each client
                myclients.clients_set[clients].local_parameters = Fed_avg_model.state_dict()
                # in the first round, calculate the summation of global_mapping parameters in init_global_mapping_models
                client_mapping_parameters = init_global_mapping_model.global_mapping.state_dict()
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in client_mapping_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += client_mapping_parameters[var]

            for var in global_mapping_parameters:
                global_mapping_parameters[var] = (sum_parameters[var] / client_num)

        # update Fed_avg_model with a specific proportion of clients
        # load global_mapping_parameters and client's private parameters -> training -> save Fed_avg_model
        # -> calculate global_mapping_parameters
        else:
            order = np.random.permutation(client_num)
            # select clients to be trained
            clients_in_comm = ['client{}'.format(i) for i in order[0:int(client_num*client_per)]]
            for clients in clients_in_comm:
                # in the second round, load local_parameters and global_mapping_parameters to each client
                Fed_avg_model.load_state_dict(myclients.clients_set[clients].local_parameters, strict=True)
                Fed_avg_model.global_mapping.load_state_dict(global_mapping_parameters, strict=True)
                myclients.clients_set[clients].localUpdate_without_personalization(local_epoch, local_batch, Fed_avg_model,
                                                           torch.optim.Adam(Fed_avg_model.parameters(), lr=0.0001),
                                                           alpha, beta)

            # collect all the local_align_mapping parameters to calculate global_mapping parameters
            val_loss = 0
            for clients in ['client{}'.format(i) for i in range(client_num)]:
                Fed_avg_model.load_state_dict(myclients.clients_set[clients].local_parameters, strict=True)
                val_loss += myclients.clients_set[clients].local_val(Fed_avg_model, alpha, beta, if_local_mapping=False, if_personalization=True)
                client_mapping_parameters = Fed_avg_model.local_align_mapping.state_dict()
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in client_mapping_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += client_mapping_parameters[var]

            for var in global_mapping_parameters:
                global_mapping_parameters[var] = (sum_parameters[var] / client_num)

            val_loss /= client_num

            # distribute global_parameters to all clients to uniform individual models
            # if val_loss does not decrease then early stopping and save the best individual models
            earlystopping(val_loss, Fed_avg_model, myclients, client_per, global_mapping_parameters, alpha, beta)
            print(val_loss)
            writer.writerow([float(val_loss)])
            if earlystopping.early_stop:
                time_end = time.time()
                print('time cost: ', time_end-time_start)
                print('communication Early stopping')
                break

def nbeats(client_set, dev=args['device'], client_num=20,
                     embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1, window_width=24,
                    local_epoch=200, local_batch=256):
    same_seeds()
    np.random.seed(6)

    # generate all the clients set
    #myclients = client_group(client_num, dev)
    myclients = client_set

    time_start = time.time()

    # get input size
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    # build the models
    init_global_mapping_model = NBEATS().to(dev)
    init_parameters = init_global_mapping_model.state_dict()

    clients_in_comm = ['client{}'.format(i) for i in range(client_num)]
    for clients in clients_in_comm:
        init_global_mapping_model.load_state_dict(init_parameters, strict=True)
        myclients.clients_set[clients].localInitialize(local_epoch, local_batch, init_global_mapping_model,
                                                       torch.optim.Adam(init_global_mapping_model
                                                                        .parameters(), lr=0.001), pa=50)
        torch.save(init_global_mapping_model, 'model_parameters_storage/nbeats/nbeats_model_{}'
                   .format(clients))
    time_end = time.time()
    print('time cost: ', time_end-time_start)

def impactnet_train(client_set, dev=args['device'], client_num=20,
                     embedding_size=20, hidden_size=64, num_layers=2, forecast_period=1, window_width=24,
                    local_epoch=200, local_batch=64):
    same_seeds()
    np.random.seed(6)

    # generate all the clients set
    #myclients = client_group(client_num, dev)
    myclients = client_set

    time_start = time.time()

    # get input size
    input_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[0].shape[1]
    feature_size = myclients.clients_set['client0'].train_ds.__getitem__(0)[1].shape[0]

    # build the models
    init_global_mapping_model = impactnet(feature_size=feature_size).to(dev)
    init_parameters = init_global_mapping_model.state_dict()

    clients_in_comm = ['client{}'.format(i) for i in range(client_num)]
    for clients in clients_in_comm:
        init_global_mapping_model.load_state_dict(init_parameters, strict=True)
        myclients.clients_set[clients].localInitialize(local_epoch, local_batch, init_global_mapping_model,
                                                       torch.optim.Adam(init_global_mapping_model
                                                                        .parameters(), lr=0.0005), pa=10)
        torch.save(init_global_mapping_model, 'model_parameters_storage/impactnet/impactnet_model_{}'
                   .format(clients))
    time_end = time.time()
    print('time cost: ', time_end-time_start)

#Fedavg_server(client_num=30, client_per=1, comm_rounds=1)
#only_Fed(client_num=1, client_per=1, comm_rounds=2)
