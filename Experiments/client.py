import numpy as np
import os
import time

#"1, 2"表示训练的时候选用两块GPU，优先选用"1"号GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import torch
from torch.utils.data import DataLoader
from data_generation import London_sm_data
from Forecasting_Models import mmd, diff_loss, Fed_avg, init_global_mapping, mae
import torch.nn as nn
import matplotlib.pyplot as plt
from pandas import read_csv
from experimental_parameters import args
import copy

# Device configuration
device = args['device']

# inverse min-max scaler
hh_load = read_csv('clear_hh_information.csv', engine='python')
load_max = np.max(hh_load['load'])
load_min = np.min(hh_load['load'])

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
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

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'Training EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        #if self.verbose:
            #print(f'Training Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), 'checkpoint.pt')
        self.best_parameters = model.state_dict().copy()
        self.val_loss_min = val_loss



def same_seeds(seed=6):
    #torch.cuda.set_device(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class client(object):
    def __init__(self, client_number, forecast_period, window_width, dev=args['device']):
        self.train_ds = London_sm_data(client=client_number, forecast_period=forecast_period, dataset_type='train',\
                                       window_width=window_width)
        self.valid_ds = London_sm_data(client=client_number, forecast_period=forecast_period, dataset_type='valid',\
                                       window_width=window_width)
        self.test_ds = London_sm_data(client=client_number, forecast_period=forecast_period, dataset_type='test',\
                                       window_width=window_width)
        self.dev = dev
        self.train_dl = None
        self.valid_dl = None
        self.test_dl = None
        self.local_parameters = None

    def localInitialize(self, localEpoch, localBatchSize, Net, opti, pa=10):
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=False, drop_last=True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=localBatchSize, shuffle=False, drop_last=True)
        early_stopping = EarlyStopping(patience=pa, verbose=True)
        Net.train()
        same_seeds()
        local_start = time.time()
        for epoch in range(localEpoch):
            #print(epoch)
            Net.train()
            for data, sf, label in self.train_dl:
                data, sf, label = data.to(self.dev), sf.to(self.dev), label.to(self.dev)
                #data, label = data.to(self.dev), label.to(self.dev)
                #print(data.shape)
                #print(label.shape)
                out = Net(data, sf)
                #print(out.shape)
                loss = nn.L1Loss()(out, label)
                #loss = nn.SmoothL1Loss()(out, label)
                #loss = nn.MSELoss()(out, label)
                #loss = (nn.MSELoss()(out, label)) ** 0.5
                #print(loss)

                loss.backward()
                opti.step()
                opti.zero_grad()


            Net.eval()
            val_loss = 0
            for data, sf, label in self.valid_dl:
                data, sf, label = data.to(self.dev), sf.to(self.dev), label.to(self.dev)
                #data, label = data.to(self.dev), label.to(self.dev)
                with torch.no_grad():
                    out = Net(data, sf)
                    val_loss += nn.L1Loss()(out, label)
                    #val_loss += nn.SmoothL1Loss()(out, label)
                    #val_loss += nn.MSELoss()(out, label)
                    #val_loss += (nn.MSELoss()(out, label))**0.5

            #val_loss /= localBatchSize
            early_stopping(val_loss, Net)
            # early stopping
            if early_stopping.early_stop:
                print("training Early stopping")
                break

        local_end = time.time()
        print('local cost:', local_end - local_start)
        # load best model
        #Net.load_state_dict(torch.load('checkpoint.pt'))
        Net.load_state_dict(early_stopping.best_parameters)

        # update self.parameters
        self.local_parameters = Net.state_dict()


    def localUpdate(self, localEpoch, localBatchSize, Net, opti, alpha=0.2, beta=1):
        #Net.load_state_dict(global_parameters, strict=True)

        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=False)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=localBatchSize, shuffle=False)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        Net.train()
        same_seeds()
        local_start = time.time()
        for epoch in range(localEpoch):
            #print(epoch)
            Net.train()
            for data, sf, label in self.train_dl:
                data, sf, label = data.to(self.dev), sf.to(self.dev), label.to(self.dev)
                h1, h2, h3, out1, out2, out3, out = Net(data, sf)
                #loss = alpha*mmd(h1, h2) + nn.L1Loss()(out, label) + beta*diff_loss(h2, h3)
                #loss = alpha * mmd(h1, h2) + (nn.MSELoss()(out, label))**0.5 + beta * diff_loss(h1, h3)
                loss = alpha * mmd(h1, h2) + nn.L1Loss()(out, label) + \
                       beta * diff_loss(h2, h3) + nn.L1Loss()(out1, label)\
                       + nn.L1Loss()(out2, label) + nn.L1Loss()(out3, label)
                #print(nn.L1Loss()(out, label))
                #print(mmd(h1, h2))
                #print(diff_loss(h2, h3))
                loss.backward()
                opti.step()
                opti.zero_grad()

            Net.eval()
            val_loss = 0
            for data, sf, label in self.valid_dl:
                data, sf, label = data.to(self.dev), sf.to(self.dev), label.to(self.dev)
                with torch.no_grad():
                    h1, h2, h3, out1, out2, out3, out = Net(data, sf)
                    #val_loss += alpha*mmd(h1, h2) + nn.L1Loss()(out, label) + beta*diff_loss(h2, h3)
                    #val_loss += alpha * mmd(h1, h2) + (nn.MSELoss()(out, label))**0.5 + beta * diff_loss(h1, h3)
                    val_loss += alpha * mmd(h1, h2) + nn.L1Loss()(out, label) + \
                            + nn.L1Loss()(out1, label) + beta*diff_loss(h2, h3) \
                           + nn.L1Loss()(out2, label) + nn.L1Loss()(out3, label)
            early_stopping(val_loss, Net)
            # early stopping
            if early_stopping.early_stop:
                print("training Early stopping")
                break

        local_end = time.time()
        print('local cost:', local_end - local_start)
        # load best model
        #Net.load_state_dict(torch.load('checkpoint.pt'))
        Net.load_state_dict(early_stopping.best_parameters)

        # update self.parameters
        self.local_parameters = Net.state_dict()






    def localUpdate_without_personalization(self, localEpoch, localBatchSize, Net, opti, alpha=0.2, beta=1):
        #Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=False)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=localBatchSize, shuffle=False)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        Net.train()
        same_seeds()
        for epoch in range(localEpoch):
            #print(epoch)
            Net.train()
            for data, sf, label in self.train_dl:
                data, sf, label = data.to(self.dev), sf.to(self.dev), label.to(self.dev)
                h1, h2, h3, out1, out2 = Net(data, sf)
                #loss = alpha*mmd(h1, h2) + nn.L1Loss()(out, label) + beta*diff_loss(h2, h3)
                #loss = alpha * mmd(h1, h2) + (nn.MSELoss()(out, label))**0.5 + beta * diff_loss(h1, h3)
                loss = alpha * mmd(h1, h2) + nn.L1Loss()(out2, label) + \
                       beta * diff_loss(h2, h3) + nn.L1Loss()(out1, label)
                #print(nn.L1Loss()(out, label))
                #print(mmd(h1, h2))
                #print(diff_loss(h2, h3))
                loss.backward()
                opti.step()
                opti.zero_grad()

            Net.eval()
            val_loss = 0
            for data, sf, label in self.valid_dl:
                data, sf, label = data.to(self.dev), sf.to(self.dev), label.to(self.dev)
                with torch.no_grad():
                    h1, h2, h3, out1, out2 = Net(data, sf)
                    #val_loss += alpha*mmd(h1, h2) + nn.L1Loss()(out, label) + beta*diff_loss(h2, h3)
                    #val_loss += alpha * mmd(h1, h2) + (nn.MSELoss()(out, label))**0.5 + beta * diff_loss(h1, h3)
                    val_loss += alpha * mmd(h1, h2) + nn.L1Loss()(out2, label) + \
                       beta * diff_loss(h2, h3) + nn.L1Loss()(out1, label)
            early_stopping(val_loss, Net)
            # early stopping
            if early_stopping.early_stop:
                print("training Early stopping")
                break
        # load best model
        #Net.load_state_dict(torch.load('checkpoint.pt'))
        Net.load_state_dict(early_stopping.best_parameters)

        # update self.parameters
        self.local_parameters = Net.state_dict()


    def local_val(self, Net, alpha=0.2, beta=1, if_local_mapping=True, if_personalization=False):
        self.valid_dl = DataLoader(self.valid_ds, batch_size=1, shuffle=False)
        Net.eval()
        i = 0
        val_loss = 0
        MMD = 0
        DIFF = 0
        for data, sf, label in self.valid_dl:
            i += 1
            data, sf, label = data.to(self.dev), sf.to(self.dev), label.to(self.dev)
            if if_local_mapping:
                with torch.no_grad():
                    h1, h2, h3, out1, out2, out3, out = Net(data, sf)
                    #val_loss += alpha * mmd(h1, h2) + nn.L1Loss()(out, label) + beta*diff_loss(h2, h3)
                    #val_loss += alpha * mmd(h1, h2) + nn.L1Loss()(out, label)
                    val_loss += nn.L1Loss()(out, label)
                    #MMD += mmd(h1, h2)
                    #DIFF += beta*diff_loss(h2, h3)


                    #val_loss += (nn.MSELoss()(out, label))**0.5
            elif if_local_mapping==False and if_personalization==True:
                with torch.no_grad():
                    h1, h2, h3, out2, out1 = Net(data, sf)
                    #val_loss += alpha * mmd(h1, h2) + nn.L1Loss()(out, label) + beta*diff_loss(h2, h3)
                    #val_loss += alpha * mmd(h1, h2) + nn.L1Loss()(out, label)
                    val_loss += nn.L1Loss()(out1, label)
                    #val_loss += (nn.MSELoss()(out, label))**0.5
            else:
                with torch.no_grad():
                    out = Net(data, sf)
                    val_loss += nn.L1Loss()(out, label)
                    #val_loss += (nn.MSELoss()(out, label))**0.5
        val_loss /= i
        MMD /= i
        DIFF /= i
        if if_local_mapping:
            return val_loss, MMD, DIFF

        return val_loss


    def local_test(self, Net, if_global_mapping = True, title='FedAVG', if_return_list=False):
        test_start = time.time()
        test_length = 720
        data, sf, label = self.test_ds.get_test_data()
        #data, sf, label = data.to(self.dev), sf.to(self.dev), label.to(self.dev)
        Net.eval()
        data = torch.tensor([item.cpu().detach().numpy() for item in data])
        sf = torch.tensor([item.cpu().detach().numpy() for item in sf])
        true_load = torch.tensor([item.cpu().detach().numpy() for item in label]).view(-1).tolist()
        predict_load = true_load.copy()[:test_length]
        for i in range(test_length):
            with torch.no_grad():
                net_out = Net(data[i:i+1].to(args['device']), sf[i:i+1].to(args['device']))
                if if_global_mapping:
                    predict_load[i] = net_out[-1][0][0]
                    if predict_load[i]<0:
                        predict_load[i]=torch.tensor(0).to(args['device'])
                    #print(predict_load[i])
                else:
                    predict_load[i] = net_out[0][0]
                    if predict_load[i]<0:
                        predict_load[i]=torch.tensor(0).to(args['device'])

        predict_load = torch.stack(predict_load).cpu().tolist()
        test_mae, test_mse, test_mape = 0, 0, 0
        predict_load = (predict_load + load_min) * (load_max - load_min)
        true_load = (true_load + load_min) * (load_max - load_min)

        if if_return_list:
            return true_load, predict_load

        for i in range(test_length):
            #print(predict_load[i])
            #print(true_load[i])
            test_mae += abs(predict_load[i]-true_load[i])
            test_mse += (predict_load[i]-true_load[i])**2
            test_mape += abs(predict_load[i]-true_load[i])/abs(true_load[i])
        test_mae /= test_length
        test_mse = (test_mse/test_length)**0.5
        test_mape /= test_length

        #print(test_mae)
        #plt.plot(true_load[:7*24], label='true')
        #plt.plot(predict_load[:7*24], label='predict')
        #plt.plot([abs(predict_load[j]-true_load[j]) for j in range(24*7)], label='predict')
        #plt.title(title)
        #plt.legend()
        #plt.xlabel('mae: '+ str(test_mae))
        #plt.show()
        test_end = time.time()
        #print('test cost:', test_end-test_start)

        return test_mae, test_mse, test_mape

    def local_persist(self):
        data, sf, label = self.test_ds.get_test_data()
        true_load = torch.tensor([item.cpu().detach().numpy() for item in label]).view(-1).tolist()
        test_mae, test_mse, test_mape = 0, 0, 0
        true_load = (true_load + load_min) * (load_max - load_min)
        test_length = 720
        for i in range(test_length):
            # print(predict_load[i])
            # print(true_load[i])
            test_mae += abs(true_load[i] - true_load[i+1])
            test_mse += (true_load[i] - true_load[i+1]) ** 2
            test_mape += abs(true_load[i] - true_load[i+1]) / abs(true_load[i+1])

        test_mae /= test_length
        test_mse = (test_mse / test_length) ** 0.5
        test_mape /= test_length

        return test_mae, test_mse, test_mape



class client_group(object):
    def __init__(self, numOfClients, dev=args['device'], forecast_period=1, window_width = 144):
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.window_width = window_width
        self.forecast_period = forecast_period

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        for i in range(self.num_of_clients):
            print(i)
            someone = client(i, self.forecast_period, self.window_width, dev=self.dev)
            self.clients_set['client{}'.format(i)] = someone

def test():
    forecast_p = 1
    window = 144
    client_1 = client(client_number=2, dev=device, forecast_period=forecast_p, window_width=window)
    # print(client_1.train_ds.__getitem__(10)[0].shape[1])
    #Net = init_global_mapping(input_size=client_1.train_ds.__getitem__(10)[0].shape[1],
     #                         feature_size=client_1.train_ds.__getitem__(10)[1].shape[0],
      #                        num_labels=forecast_p, hidden_size=100)
    Net = Fed_avg(input_size=client_1.train_ds.__getitem__(10)[0].shape[1],
                              feature_size=client_1.train_ds.__getitem__(10)[1].shape[0],
                              num_labels=forecast_p, hidden_size=64)
    Net = Net.to(device)
    print(Net)
    #client_1.localInitialize(20, 64, Net, torch.optim.Adam(Net.parameters(), lr=0.001))
    client_1.localUpdate(20, 64, Net, torch.optim.Adam(Net.parameters(), lr=0.001))
    client_1.local_test(Net)

#test()