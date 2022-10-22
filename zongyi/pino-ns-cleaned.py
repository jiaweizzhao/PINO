import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities4 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

import wandb

import argparse
parser = argparse.ArgumentParser(description='Incremental PINO')

parser.add_argument('--name', type=str, default='test')
parser.add_argument('--method', type=str, default='standard')
parser.add_argument('--max_modes', default=8, type=int)
parser.add_argument('--init_modes', default=1, type=int)

# for loss gap method
parser.add_argument('--loss_eps', default=1e-5, type=float)

args = parser.parse_args()

wandb.init(project="incremental-fno", entity="jiawei", name=args.name)
wandb.config.update(args)

torch.manual_seed(0)
np.random.seed(0)

activation = F.relu


def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return op(a, b)
    # op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    # return torch.stack([
    #     op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
    #     op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    # ], dim=-1)

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.max_modes = modes1 # assert modes1 == modes2 == modes3

        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        
        # reassign the modes
        self.adaptive_modes1 = args.init_modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.adaptive_modes2 = args.init_modes 
        self.adaptive_modes3 = args.init_modes 

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3,4], norm="ortho")
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.adaptive_modes1, :self.adaptive_modes2, :self.adaptive_modes3] = \
            compl_mul3d(x_ft[:, :, :self.adaptive_modes1, :self.adaptive_modes2, :self.adaptive_modes3], self.weights1[:,:,:self.adaptive_modes1, :self.adaptive_modes2, :self.adaptive_modes3])
        out_ft[:, :, -self.adaptive_modes1:, :self.adaptive_modes2, :self.adaptive_modes3] = \
            compl_mul3d(x_ft[:, :, -self.adaptive_modes1:, :self.adaptive_modes2, :self.adaptive_modes3], self.weights2[:,:,:self.adaptive_modes1, :self.adaptive_modes2, :self.adaptive_modes3])
        out_ft[:, :, :self.adaptive_modes1, -self.adaptive_modes2:, :self.adaptive_modes3] = \
            compl_mul3d(x_ft[:, :, :self.adaptive_modes1, -self.adaptive_modes2:, :self.adaptive_modes3], self.weights3[:,:,:self.adaptive_modes1, :self.adaptive_modes2, :self.adaptive_modes3])
        out_ft[:, :, -self.adaptive_modes1:, -self.adaptive_modes2:, :self.adaptive_modes3] = \
            compl_mul3d(x_ft[:, :, -self.adaptive_modes1:, -self.adaptive_modes2:, :self.adaptive_modes3], self.weights4[:,:,:self.adaptive_modes1, :self.adaptive_modes2, :self.adaptive_modes3])

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2,3,4], norm="ortho")
        return x
    
    def determine_modes(self, ep, loss):
        
        if args.method == 'standard':
            self.adaptive_modes1 = self.max_modes
            self.adaptive_modes2 = self.max_modes
            self.adaptive_modes3 = self.max_modes
        elif args.method == 'loss_gap':
            eps = 1e-5
            # method 1: loss_gap
            if not hasattr(self, 'loss_list'):
                self.loss_list = [loss]
            else:
                self.loss_list.append(loss)
            
            if len(self.loss_list) > 1:
                if abs(self.loss_list[-1] - self.loss_list[-2]) <= eps:
                    if self.adaptive_modes1 < self.max_modes:
                        self.adaptive_modes1 += 1
                    if self.adaptive_modes2 < self.max_modes:
                        self.adaptive_modes2 += 1
                    if self.adaptive_modes3 < self.max_modes:
                        self.adaptive_modes3 += 1
        elif args.method == 'frequency_norm':
            pass
            
            
        # log mode changes
        # print('modes1: {}, modes2: {}, modes3: {}'.format(self.adaptive_modes1, self.adaptive_modes2, self.adaptive_modes3))
        wandb.log({'modes1': self.adaptive_modes1, 'modes2': self.adaptive_modes2, 'modes3': self.adaptive_modes3, 'epoch': ep})
        
    

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(4, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        x0 = x[..., -1:]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = torch.tanh(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = torch.tanh(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = torch.tanh(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x #+ x0

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes, modes, width)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
    
    def determine_modes(self, ep, loss):
        self.conv1.conv0.determine_modes(ep, loss)
        self.conv1.conv1.determine_modes(ep, loss)
        self.conv1.conv2.determine_modes(ep, loss)
        self.conv1.conv3.determine_modes(ep, loss)

def get_forcing(S):
    x1 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(S, 1).repeat(1, S)
    x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
    return -4 * (torch.cos(4*(x2))).reshape(1,S,S,1).cuda()

def pad_grid(data, S, T):
    gridx = torch.tensor(np.linspace(0, 1, S+1)[:-1], dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S+1)[:-1], dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    n = data.shape[0]
    return torch.cat((gridx.repeat([n,1,1,1,1]), gridy.repeat([n,1,1,1,1]),
                       gridt.repeat([n,1,1,1,1]), data), dim=-1)

def FDM_NS_vorticity(w, v=1 / 500, T_interval=1.0):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    wx = torch.fft.irfft2(wx_h[:, :, :k_max + 1], dim=[1, 2])
    wy = torch.fft.irfft2(wy_h[:, :, :k_max + 1], dim=[1, 2])
    wlap = torch.fft.irfft2(wlap_h[:, :, :k_max + 1], dim=[1, 2])

    dt = T_interval / (nt)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    Du1 = wt + (ux * wx + uy * wy - v * wlap)[..., 1:-1]  # - forcing
    return Du1

def PINO_loss(u, u0, forcing, T_interval=1.0):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    Du = FDM_NS_vorticity(u, T_interval=T_interval)
    f = forcing.repeat(batch_size, 1, 1, nt - 2)
    loss_f = lploss(Du, f)

    return loss_ic, loss_f

Ntest = 1
ntest = Ntest

modes = 8
width = 64

batch_size = 1
batch_size2 = batch_size
device = torch.device('cuda')

epochs = 5000
learning_rate = 0.0025
scheduler_step = 1000
scheduler_gamma = 0.5
print(epochs, learning_rate, scheduler_step, scheduler_gamma)


index = 0
sub_test = 4
ns = 256 // sub_test
# nt = 64
T_pad = 11
T_interval = 0.5
nt = 64
T_index = int(T_interval*128)
t1_full = default_timer()

HOME_PATH = '/ngc_workspace/project/PINO/'
path = 'pino_fdm_ns500_'+str(T_interval)+'s_N'+str(ntest)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width) + '_s' + str(ns) + '_t' + str(nt) + '_T' + str(T_interval) + '_i' + str(index)
path_model = HOME_PATH+'model/'+path
path_train_err = HOME_PATH+'results/'+path+'train.txt'
path_test_err = HOME_PATH+'results/'+path+'test.txt'
path_image = HOME_PATH+'image/'+path

data = np.load(HOME_PATH+'data/NS_Re500_s256_T100_test.npy')
print(data.shape)

test_a = torch.tensor(data, dtype=torch.float)[index,0,::sub_test,::sub_test].reshape(1, ns, ns, 1)
test_u = torch.tensor(data, dtype=torch.float)[index,T_index,::sub_test,::sub_test].reshape(1, ns, ns, 1)

print(torch.mean(torch.abs(test_a)), torch.mean(torch.abs(test_u)))
test_a = test_a.reshape(ntest, ns, ns, 1, 1).repeat([1,1,1,nt,1])

test_a = pad_grid(test_a, ns, nt)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

forcing_train = get_forcing(ns)
forcing_test = get_forcing(ns)

myloss = LpLoss(size_average=True)
error = np.zeros((epochs, 4))

model = Net2d(args.max_modes, width).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# wandb.config = {

# }
# wandb.watch(model)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    test_pino = 0.0
    test_l2 = 0.0
    test_l2_T = 0.0
    test_f = 0.0

    for x, y in test_loader: # only one data point
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()

        x_in = F.pad(x, (0,0,0,T_pad), "constant", 0)
        out = model(x_in.reshape(batch_size,ns,ns,nt+T_pad,4)).reshape(batch_size,ns,ns,nt+T_pad)
        out = out[..., :-T_pad]
        # out = model(x).reshape(batch_size,S,S,T)
        x_init = x[:, :, :, 0, -1]

        loss_ic, loss_f = PINO_loss(out.view(batch_size, ns, ns, nt), x_init, forcing_test, T_interval=T_interval)
        pino_loss = (loss_ic*10 + loss_f)

        pino_loss.backward()
        optimizer.step()
        # test_l2 = loss.item()
        test_pino = pino_loss.item()
        test_f = loss_f.item()
        test_l2_T = myloss(out.view(batch_size, ns, ns, nt)[..., -1], y.view(batch_size, ns, ns)).item()
        loss_ic = loss_ic.item()
        

    scheduler.step()
    t2 = default_timer()
    print(ep, t2-t1, test_pino, loss_ic, test_f, test_l2, test_l2_T) #test_l2_u)
    wandb.log({'test_pino': test_pino, 'loss_ic': loss_ic, 'loss_f': test_f, 'loss_l2': test_l2, 'loss_l2_T': test_l2_T,'epoch': ep})

    model.determine_modes(ep, test_l2)

    # if ep % 1000 == 1:
    #     y = y[0,:,:,:].cpu().numpy()
    #     out = out[0,:,:,:].detach().cpu().numpy()
    #
    #     fig, ax = plt.subplots(3, 5)
    #     ax[0,0].imshow(y[..., 0])
    #     ax[0,1].imshow(y[..., 16])
    #     ax[0,2].imshow(y[..., 32])
    #     ax[0,3].imshow(y[..., 48])
    #     ax[0,4].imshow(y[..., 64])
    #
    #     ax[1,0].imshow(out[..., 0])
    #     ax[1,1].imshow(out[..., 16])
    #     ax[1,2].imshow(out[..., 32])
    #     ax[1,3].imshow(out[..., 48])
    #     ax[1,4].imshow(out[..., 64])
    #
    #     ax[2,0].imshow(y[..., 0]-out[..., 0])
    #     ax[2,1].imshow(y[..., 16]-out[..., 16])
    #     ax[2,2].imshow(y[..., 32]-out[..., 32])
    #     ax[2,3].imshow(y[..., 48]-out[..., 48])
    #     ax[2,4].imshow(y[..., 64]-out[..., 64])
    #     plt.show()



# torch.save(model, path_model+str(T_interval)+'_finetune0')
t2_full = default_timer()
print("finished", ns, nt, T_interval, index, ep, test_l2_T)
print(ep, t2_full - t1_full, test_pino, test_f, test_l2, test_l2_T)  # test_l2_u)


# outu = w_to_u(out.view(S_test, S_test, T).permute(2,0,1))
# yu = w_to_u(y.view(S_test, S_test, T).permute(2,0,1))
# scipy.io.savemat(HOME_PATH+'pred/'+path+'.mat', mdict={'pred': out.detach().cpu().numpy(), 'truth': y.detach().cpu().numpy(),'predu': outu.detach().cpu().numpy(), 'truthu': yu.detach().cpu().numpy()})


