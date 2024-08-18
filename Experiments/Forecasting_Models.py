import os

#"1, 2"表示训练的时候选用两块GPU，优先选用"1"号GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import math
from experimental_parameters import args



# Device configuration
device = args['device']

class global_mapping(nn.Module):
    def __init__(self, input_size, feature_size=1, embedding_size=74, hidden_size=64, num_layers=1, window_width=24):
        super(global_mapping, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_layer = nn.Linear(input_size, embedding_size, bias=False)
        self.feature_size = feature_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        # self.lstmfc1 = nn.Linear(2*window_width, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.res_fc3 = nn.Linear(self.feature_size, hidden_size)

    def forward(self, x_temporal, x_nt):
        # Set initial hidden and cell states
        # ed = self.embedding_layer(x_temporal)
        # ed = nn.Tanh()(ed)

        st3 = self.res_fc3(x_nt)
        st = nn.ELU()(st3)
        st = self.fc1(st)
        st = nn.ELU()(st)
        st = self.fc2(st)
        st = nn.ELU()(st)

        # Forward propagate LSTM
        hd, _ = self.lstm(x_temporal)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # hd = self.lstmfc1(torch.flatten(x_temporal, 1))
        hd = (hd[:, -1, :])
        #hd = self.fc3(hd)
        #hd = (hd)
        # hidden_state = torch.cat((hd[:, -1, :], st), dim=-1)
        hidden_state = torch.cat((hd, st), dim=-1)
        hidden_state = self.fc4(hidden_state)

        # Decode the hidden state of the last time step
        return hidden_state

class local_align_mapping(nn.Module):
    def __init__(self, input_size, feature_size=1, embedding_size=74, hidden_size=64, num_layers=1, window_width=24):
        super(local_align_mapping, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_layer = nn.Linear(input_size, embedding_size, bias=False)
        self.feature_size = feature_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        # self.lstmfc1 = nn.Linear(2*window_width, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.res_fc3 = nn.Linear(self.feature_size, hidden_size)

    def forward(self, x_temporal, x_nt):
        # Set initial hidden and cell states
        # ed = self.embedding_layer(x_temporal)
        # ed = nn.Tanh()(ed)

        st3 = self.res_fc3(x_nt)
        st = nn.ELU()(st3)
        st = self.fc1(st)
        st = nn.ELU()(st)
        st = self.fc2(st)
        st = nn.ELU()(st)

        # Forward propagate LSTM
        hd, _ = self.lstm(x_temporal)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # hd = self.lstmfc1(torch.flatten(x_temporal, 1))
        hd = (hd[:, -1, :])
        #hd = self.fc3(hd)
        #hd = (hd)
        # hidden_state = torch.cat((hd[:, -1, :], st), dim=-1)
        hidden_state = torch.cat((hd, st), dim=-1)
        hidden_state = self.fc4(hidden_state)

        # Decode the hidden state of the last time step
        return hidden_state

class local_orthogonal_mapping(nn.Module):
    def __init__(self, input_size, feature_size=1, embedding_size=74, hidden_size=64, num_layers=1, window_width=24):
        super(local_orthogonal_mapping, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_layer = nn.Linear(input_size, embedding_size, bias=False)
        self.feature_size = feature_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        # self.lstmfc1 = nn.Linear(2*window_width, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.res_fc3 = nn.Linear(self.feature_size, hidden_size)

    def forward(self, x_temporal, x_nt):
        # Set initial hidden and cell states
        # ed = self.embedding_layer(x_temporal)
        # ed = nn.Tanh()(ed)

        st3 = self.res_fc3(x_nt)
        st = nn.ELU()(st3)
        st = self.fc1(st)
        st = nn.ELU()(st)
        st = self.fc2(st)
        st = nn.ELU()(st)

        # Forward propagate LSTM
        hd, _ = self.lstm(x_temporal)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # hd = self.lstmfc1(torch.flatten(x_temporal, 1))
        hd = (hd[:, -1, :])
        #hd = self.fc3(hd)
        #hd = (hd)
        # hidden_state = torch.cat((hd[:, -1, :], st), dim=-1)
        hidden_state = torch.cat((hd, st), dim=-1)
        hidden_state = self.fc4(hidden_state)

        # Decode the hidden state of the last time step
        return hidden_state


class Fed_avg(nn.Module):
    def __init__(self, input_size, feature_size, embedding_size=10, hidden_size=128, num_layers=1, num_labels=24, window_width=24):
        super(Fed_avg, self).__init__()
        self.global_mapping = global_mapping(input_size, feature_size, embedding_size, hidden_size, num_layers, window_width)
        self.local_align_mapping = local_align_mapping(input_size, feature_size, embedding_size, hidden_size, num_layers, window_width)
        self.local_orthogonal_mapping = local_orthogonal_mapping(input_size, feature_size, embedding_size, hidden_size, num_layers, window_width)
        self.local_model = local_orthogonal_mapping(input_size, feature_size, embedding_size, hidden_size,
                                                                 num_layers, window_width)

        self.outfc1 = nn.Linear(hidden_size, num_labels)
        self.outfc2 = nn.Linear(hidden_size, num_labels)
        self.outfc3 = nn.Linear(hidden_size, num_labels)
        #self.dcfc1 = nn.Linear(window_width*2, hidden_size)
        self.dcfc1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=False)
        self.dcfc2 = nn.Linear(feature_size, hidden_size)
        self.dcfc3 = nn.Linear(hidden_size*2, 1)

    def forward(self, x_temporal, x_nt):
        # decision-making
        #dc1 = self.dcfc1(torch.flatten(x_temporal, 1))
        dc1, _ = self.dcfc1(x_temporal)
        #dc1 = nn.ELU()(dc1)
        dc2 = self.dcfc2(x_nt)
        #dc2 = nn.ELU()(dc2)
        dc3 = self.dcfc3(torch.cat((dc1[:,-1,:], dc2), dim=1))
        dc3 = nn.Sigmoid()(dc3)
        one_tensor = torch.ones(size=[x_temporal.shape[0], 1]).to(device)

        h1 = self.global_mapping(x_temporal, x_nt)
        h2 = self.local_align_mapping(x_temporal, x_nt)
        h3 = self.local_orthogonal_mapping(x_temporal, x_nt)
        h4 = self.local_model(x_temporal, x_nt)
        out1 = self.outfc1(h2)
        out2 = self.outfc2(h2+h3)
        out3 = self.outfc3(h4)
        r1 = torch.mul(out1, one_tensor-dc3)
        r2 = torch.mul(out3, dc3)


        out = r1+r2


        # Decode the hidden state of the last time step
        return h1, h2, h3, out1, out2, out3, out

class init_global_mapping(nn.Module):
    def __init__(self, input_size, feature_size, embedding_size=20, hidden_size=100, num_layers=2, num_labels=24, window_width=24):
        super(init_global_mapping, self).__init__()
        self.global_mapping = global_mapping(input_size, feature_size, embedding_size, hidden_size, num_layers, window_width)
        #self.fc1 = nn.Linear(hidden_size, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        #self.fc3 = nn.Linear(hidden_size // 2, num_labels)
        self.fco = nn.Linear(hidden_size, num_labels)

    def forward(self, x_temporal, st):
        # Set initial hidden and cell states
        hd = self.global_mapping(x_temporal, st)
        out = self.fco(hd)
        #hd = self.fc1(hd)
        #hd = nn.ELU()(hd)
        #out = self.fc2(hd)
        #out = nn.ELU()(out)
        #out = self.fc3(out)

        # Decode the hidden state of the last time step
        return out

class Fed_avg_without_personalization(nn.Module):
    def __init__(self, input_size, feature_size, embedding_size=10, hidden_size=128, num_layers=1, num_labels=24, window_width=24):
        super(Fed_avg_without_personalization, self).__init__()
        self.global_mapping = global_mapping(input_size, feature_size, embedding_size, hidden_size, num_layers, window_width)
        self.local_align_mapping = local_align_mapping(input_size, feature_size, embedding_size, hidden_size, num_layers, window_width)
        self.local_orthogonal_mapping = local_orthogonal_mapping(input_size, feature_size, embedding_size, hidden_size, num_layers, window_width)
        self.local_model = local_orthogonal_mapping(input_size, feature_size, embedding_size, hidden_size,
                                                                 num_layers, window_width)

        self.outfc1 = nn.Linear(hidden_size, num_labels)
        self.outfc2 = nn.Linear(hidden_size, num_labels)
        self.outfc3 = nn.Linear(hidden_size, num_labels)
        #self.dcfc1 = nn.Linear(window_width*2, hidden_size)
        self.dcfc1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=False)
        self.dcfc2 = nn.Linear(feature_size, hidden_size)
        self.dcfc3 = nn.Linear(hidden_size*2, 1)

    def forward(self, x_temporal, x_nt):
        # decision-making
        #dc1 = self.dcfc1(torch.flatten(x_temporal, 1))
        dc1, _ = self.dcfc1(x_temporal)
        #dc1 = nn.ELU()(dc1)
        dc2 = self.dcfc2(x_nt)
        #dc2 = nn.ELU()(dc2)
        dc3 = self.dcfc3(torch.cat((dc1[:,-1,:], dc2), dim=1))
        #dc3 = nn.Sigmoid()(dc3)
        #one_tensor = torch.ones(size=[x_temporal.shape[0], 1]).to(device)

        h1 = self.global_mapping(x_temporal, x_nt)
        h2 = self.local_align_mapping(x_temporal, x_nt)
        h3 = self.local_orthogonal_mapping(x_temporal, x_nt)
        #h4 = self.local_model(x_temporal, x_nt)
        out1 = self.outfc1(h2)
        out2 = self.outfc2(h2+h3)
        #out3 = self.outfc3(h4)
        #r1 = torch.mul(out1, one_tensor-dc3)
        #r2 = torch.mul(out3, dc3)


        #out = r1+r2


        # Decode the hidden state of the last time step
        return h1, h2, h3, out2, out1


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram/核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差

		return: [	K_ss K_st
				K_ts K_tt ]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    #bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    bandwidth_list = [float(i+1)**2 for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起


def mmd(source, target, kernel_mul=2, kernel_num=20, fix_sigma=5):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = torch.mean(XX + YY - XY - YX)  # 这里是假定X和Y的样本数量是相同的
    # 当不同的时候，就需要乘上上面的M矩阵
    return loss

def diff_loss(source, target):
    A = torch.matmul(source, target.permute(1, 0))
    loss = torch.norm(A, p='fro')**2/(torch.norm(source, p='fro')*torch.norm(target, p='fro'))
    #loss = 0
    #for i in range(A.shape[0]):
     #   for j in range(A.shape[1]):
      #      loss += abs(A[i][j])
    return loss

def mae(source, target):
    loss = 0
    for i in range(source.shape[0]):
        loss += abs(source[i, 0]-target[i, 0])
    return loss


class NBEATS(nn.Module):
    def __init__(self, block_num=2, stack_num=2, loopback_window=168, future_horizen=1):
        '''
        参数说明：
            loopback_window 向前看几个时间点
            future_horizen 预测未来几个时间点
        '''
        super(NBEATS, self).__init__()
        self.name = 'apps.fmml.NbeatsModel'
        self.stack_num = stack_num
        block_stacks = []
        for _ in range(self.stack_num):
            block_stacks.append(
                BlockStack(block_num, loopback_window, future_horizen)
            )
        self.block_stacks = nn.ModuleList(block_stacks)

    def forward(self, x, x_non):
        x_hat_prev = x[:,:,0]
        y_sum = None
        x_hat = None
        for idx in self.block_stacks:
            y_hat, x_hat = idx(x_hat_prev)
            x_hat_prev = x_hat_prev-x_hat
            if y_sum is None:
                y_sum = y_hat
            else:
                y_sum = y_sum+y_hat
        return y_sum

class BasicBlock(nn.Module):
    def __init__(self, loopback_window=168, future_horizen=1):
        '''
        参数说明：
            loopback_window 向前看几个时间点
            future_horizen 预测未来几个时间点
        '''
        super(BasicBlock, self).__init__()
        self.loopback_window = loopback_window
        self.future_horizen = future_horizen
        theta_f_num = 5
        theta_b_num = 5
        # 组成块的全连接层
        fc_layers = [64, 64, 64, 64]
        self.h1 = nn.Linear(loopback_window, fc_layers[0])
        self.h2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.h3 = nn.Linear(fc_layers[1], fc_layers[2])
        self.h4 = nn.Linear(fc_layers[2], fc_layers[3])
        self.theta_f = nn.Linear(fc_layers[3], theta_f_num, bias=False)
        self.y_hat = nn.Linear(theta_f_num, future_horizen)
        #self.y_hat = nn.Linear(fc_layers[3], future_horizen, bias=False)
        self.theta_b = nn.Linear(fc_layers[3], theta_b_num, bias=False)
        self.x_hat = nn.Linear(theta_b_num, self.loopback_window)
        #self.x_hat = nn.Linear(fc_layers[3], self.loopback_window, bias=False)

    def forward(self, x):
        h1 = self.h1(x)
        h1 = nn.ReLU()(h1)
        h2 = self.h2(h1)
        h2 = nn.ReLU()(h2)
        h3 = self.h3(h2)
        h3 = nn.ReLU()(h3)
        h4 = self.h4(h3)
        h4 = nn.ReLU()(h4)
        theta_f = self.theta_f(h4)
        y_hat = self.y_hat(theta_f)
        #y_hat = self.y_hat(h4)
        theta_b = self.theta_b(h4)
        x_hat = self.x_hat(theta_b)
        #x_hat = self.x_hat(h4)
        return y_hat, x_hat

class BlockStack(nn.Module):
    def __init__(self, block_num=3, loopback_window=168, future_horizen=1):
        super(BlockStack, self).__init__()
        self.block_num = block_num
        basic_blocks = []
        for _ in range(self.block_num):
            basic_blocks.append(
                BasicBlock(loopback_window, future_horizen)
            )
        self.basic_blocks = nn.ModuleList(basic_blocks)

    def forward(self, x):
        x_hat_prev = x
        y_sum = None
        x_hat = None
        for idx in self.basic_blocks:
            y_hat, x_hat = idx(x_hat_prev)
            x_hat_prev = x_hat_prev - x_hat
            if y_sum is None:
                y_sum = y_hat
            else:
                y_sum = y_sum + y_hat
        return y_sum, x_hat


class RCU_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        '''
        参数说明：
            loopback_window 向前看几个时间点
            future_horizen 预测未来几个时间点
        '''
        super(RCU_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.relu(x)
        x1 = x
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x + x1
        x = self.relu(x)

        return x


class impactnet(nn.Module):
    def __init__(self, feature_size, out_channels=15, kernel_size=5, stride=1):
        '''
        参数说明：
            loopback_window 向前看几个时间点
            future_horizen 预测未来几个时间点
        '''
        super(impactnet, self).__init__()
        self.gru1 = RCU_block(2, out_channels, kernel_size, stride=stride)
        self.gru2 = RCU_block(out_channels, out_channels, kernel_size, stride=stride)
        self.gru3 = RCU_block(out_channels, out_channels, kernel_size, stride=stride)
        self.gru4 = RCU_block(out_channels, out_channels, kernel_size, stride=stride)
        self.max_pool = nn.MaxPool1d(kernel_size=4)
        self.relu = nn.ReLU()
        self.convfc = nn.Linear(42*out_channels, 96)
        self.fc1 = nn.Linear(feature_size, 96)
        self.fc2 = nn.Linear(96, 96)
        self.outfc1 = nn.Linear(96*2, 64)
        self.outfc2 = nn.Linear(64, 1)

    def forward(self, x, x_n):
        x = self.gru1(x.permute(0, 2, 1))
        x = self.gru2(x)
        x = self.gru3(x)
        x = self.gru4(x)
        x = self.max_pool(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = self.convfc(x)

        x_n = self.fc1(x_n)
        x_n = nn.ReLU()(x_n)
        x_n = self.fc2(x_n)
        x_n = nn.ReLU()(x_n)

        h = torch.cat((x,x_n), dim=1)
        h = self.outfc1(h)
        h = nn.ReLU()(h)
        h = self.outfc2(h)

        return h



#model = impactnet()
#inputs = torch.randn((1, 168, 2))
#print(model(inputs))

#mapping = mapping_model(10)
#g = global_model(10, 10, 1, 10, mapping)
#print(g)
#Net = init_global_mapping(input_size=74, num_labels=1)
#print(Net)