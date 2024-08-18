import argparse
import os

#"1, 2"表示训练的时候选用两块GPU，优先选用"1"号GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-cn', '--client_num', type=int, default=20,
                    help='total number of the participated clients')
parser.add_argument('-cp', '--client_per_space', type=list, default=[0.5],
                    help='space of the percentage of clients in every communication round')
parser.add_argument('-as', '--alpha_searching_space', type=list, default=[1],
                    help='searching space of the hyperparameter: alpha')
parser.add_argument('-bs', '--beta_searching_space', type=list, default=[100],
                    help='searching space of the hyperparameter: beta')
parser.add_argument('-ww', '--window_width', type=int, default=168,
                    help='width of sliding window')
parser.add_argument('-hs', '--hidden_size', type=int, default=32,
                    help='hidden size of each layer')
parser.add_argument('-nl', '--num_layers', type=int, default=1,
                    help='the number of LSTM layers')
parser.add_argument('-dv', '--device', default= torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
                    help='the device')
print(torch.cuda.is_available())
args = parser.parse_args()
args = args.__dict__