
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

import wandb

import argparse
parser = argparse.ArgumentParser(description='Incremental PINO')

parser.add_argument('--name', type=str, default='test')
parser.add_argument('--method', type=str, default='standard')
parser.add_argument('--max_modes', default=20, type=int)
parser.add_argument('--init_modes', default=1, type=int)

# for loss gap method
parser.add_argument('--loss_eps', default=1e-5, type=float)

# for frequency_norm_abs method
parser.add_argument('--norm_abs_eps', default=1.0, type=float)

# for grad_frequency_explain
parser.add_argument('--grad_max_iter', default=10, type=int)
parser.add_argument('--grad_explained_ratio_threshold', default=0.9, type=float)
parser.add_argument('--buffer', default=5, type=int)

# visualization
parser.add_argument('--visualize_evolution', action='store_true')

# resolution control
parser.add_argument('--r_data', default=42, type=int) # 42 - low, r_data = 42 or turn off data loss 
parser.add_argument('--r_pde', default=7, type=int) # fix r_pde = 7 will work (note: design a max cap for mode increaasing)

args = parser.parse_args()

wandb.init(project="incremental-fno-pino-darcy", entity="research-pino_ifno", name=args.name)
wandb.config.update(args)

torch.manual_seed(0)
np.random.seed(0)

activation = F.relu

# tools box
def compute_explained_variance(frequency_max, s):
    s_current = s.clone()
    s_current[frequency_max:] = 0
    return 1 - torch.var(s - s_current) / torch.var(s)


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)

    # return torch.stack([
    #     op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
    #     op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    # ], dim=-1)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.max_modes = modes1 # assert modes1 == modes2 == modes3
        
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        
        # reassign the modes
        if args.method == 'standard':
            self.adaptive_modes1 = self.modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1 #? to-do: set a hard limit
            self.adaptive_modes2 = self.modes2
        else:
            self.adaptive_modes1 = args.init_modes 
            self.adaptive_modes2 = args.init_modes 

    def forward(self, x, size=None):
        if size==None:
            size = x.size(-1)

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3])
        

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, size, size//2 + 1, device=x.device, dtype=torch.cfloat)
        
        # checking
        # print(x_ft.shape)
        # print(out_ft.shape)
        # print(self.weights1.shape)
        # print('after truncating')
        # print(x_ft[:, :, :self.adaptive_modes1, :self.adaptive_modes2].shape)
        # print(out_ft[:, :, :self.adaptive_modes1, :self.adaptive_modes2].shape)
        # print(self.weights1[:, :, :self.adaptive_modes1, :self.adaptive_modes2].shape)
        
        out_ft[:, :, :self.adaptive_modes1, :self.adaptive_modes2] = \
            compl_mul2d(x_ft[:, :, :self.adaptive_modes1, :self.adaptive_modes2], self.weights1[:, :, :self.adaptive_modes1, :self.adaptive_modes2])
        out_ft[:, :, -self.adaptive_modes1:, :self.adaptive_modes2] = \
            compl_mul2d(x_ft[:, :, -self.adaptive_modes1:, :self.adaptive_modes2], self.weights2[:, :, :self.adaptive_modes1, :self.adaptive_modes2])
            


        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(size, size), dim=[2,3])
        return x
    
    def determine_modes(self, ep, loss, layer_name=None):
        
        if args.method == 'standard':
            self.adaptive_modes1 = self.max_modes
            self.adaptive_modes2 = self.max_modes
        elif args.method == 'loss_gap':
            eps = args.loss_eps
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
        elif args.method == 'frequency_norm_abs':
            # using the absolute frequency norm at each mode (after weight update) to determine whether to increase
            
            # caculate the frequency norm
            weight_list = [self.weights1] #, self.weights2, self.weights3, self.weights4] #! temp version: onlt use first weight to determine
            for parameters in weight_list:
                weights = parameters.data
                # method 1: only compute the highest representable frequency mode
                # first mode direction
                strength = torch.norm(weights[:,:,self.adaptive_modes1-1,:,:], p='fro') # will be removed in the future, see documetation
                if strength >= args.norm_abs_eps:
                    if self.adaptive_modes1 < self.max_modes:
                        self.adaptive_modes1 += 1
                        print('increase mode 1')
                
                # second mode direction
                strength = torch.norm(weights[:,:,:,self.adaptive_modes2-1,:], p='fro') # will be removed in the future, see documetation
                if strength >= args.norm_abs_eps:
                    if self.adaptive_modes2 < self.max_modes:
                        self.adaptive_modes2 += 1  
                        print('increase mode 2')
                        

                # third mode direction
                strength = torch.norm(weights[:,:,:,:,self.adaptive_modes3-1], p='fro') # will be removed in the future, see documetation
                if strength >= args.norm_abs_eps:
                    if self.adaptive_modes3 < self.max_modes:
                        self.adaptive_modes3 += 1  
                        print('increase mode 3')
        
        elif args.method == 'grad_frequency_explain':                
            # method 2: explain an averaged gradient distribution more than a threshold given the current modes
            
            # average the gradient over a certain period
            if not hasattr(self, 'accumulated_grad'):
                self.accumulated_grad = torch.zeros_like(self.weights1.grad.data)
            if not hasattr(self, 'grad_iter'):
                self.grad_iter = 1
            
            if self.grad_iter <= args.grad_max_iter:
                self.grad_iter += 1
                self.accumulated_grad += self.weights1.grad.data
            
            # compute and add modes given explained variance
            else:
                # for mode 1
                weights = self.accumulated_grad
                strength_vector = []
                for mode_index in range(self.adaptive_modes1):
                    strength = torch.norm(weights[:,:,mode_index,:,:], p='fro').cpu()
                    strength_vector.append(strength)
                expained_ratio = compute_explained_variance(self.adaptive_modes1 - args.buffer, torch.Tensor(strength_vector))
                wandb.log({layer_name+'/modes1_expained_ratio': expained_ratio})
                if expained_ratio < args.grad_explained_ratio_threshold:
                    if self.adaptive_modes1 < self.max_modes:
                        self.adaptive_modes1 += 1
                        print('increase mode 1')

                # for mode 2
                weights = self.accumulated_grad
                strength_vector = []
                for mode_index in range(self.adaptive_modes2):
                    strength = torch.norm(weights[:,:,mode_index,:,:], p='fro').cpu()
                    strength_vector.append(strength)
                expained_ratio = compute_explained_variance(self.adaptive_modes2 - args.buffer, torch.Tensor(strength_vector))
                wandb.log({layer_name+'/modes2_expained_ratio': expained_ratio})
                if expained_ratio < args.grad_explained_ratio_threshold:
                    if self.adaptive_modes2 < self.max_modes:
                        self.adaptive_modes2 += 1
                        print('increase mode 2')
                        
                # for mode 3
                weights = self.accumulated_grad
                strength_vector = []
                for mode_index in range(self.adaptive_modes3):
                    strength = torch.norm(weights[:,:,mode_index,:,:], p='fro').cpu()
                    strength_vector.append(strength)
                expained_ratio = compute_explained_variance(self.adaptive_modes3 - args.buffer, torch.Tensor(strength_vector))
                wandb.log({layer_name+'/modes3_expained_ratio': expained_ratio})
                if expained_ratio < args.grad_explained_ratio_threshold:
                    if self.adaptive_modes3 < self.max_modes:
                        self.adaptive_modes3 += 1
                        print('increase mode 3')
                
                # reset
                self.grad_iter = 1
                self.accumulated_grad = torch.zeros_like(self.weights1.grad.data)
                
            
                
                
        if args.visualize_evolution:
            # visualize evolution for all modes in modes1 in weights1 in each layer
            weights = self.weights1.data
            for mode_index in range(self.max_modes):
                strength = torch.norm(weights[:,:,mode_index,:], p='fro').cpu()
                if mode_index == 0:
                    strength_list = [strength]
                else:
                    strength_list.append(strength)
            wandb.log({layer_name+'/modes1_evolution': wandb.Histogram(np.array(strength_list))})
            
            # visualize gradient 
            weights = self.weights1.grad.data
            for mode_index in range(self.max_modes):
                strength = torch.norm(weights[:,:,mode_index,:], p='fro').cpu()
                if mode_index == 0:
                    strength_list = [strength]
                else:
                    strength_list.append(strength)
            wandb.log({layer_name+'/modes1_grad_evolution': wandb.Histogram(np.array(strength_list))})            
    
            # visualize evolution for all modes in modes2 in weights1 in each layer
            weights = self.weights1.data
            for mode_index in range(self.max_modes):
                strength = torch.norm(weights[:,:,:,mode_index], p='fro').cpu()
                if mode_index == 0:
                    strength_list = [strength]
                else:
                    strength_list.append(strength)
            wandb.log({layer_name+'/modes2_evolution': wandb.Histogram(np.array(strength_list))})
            
            # visualize gradient
            weights = self.weights1.grad.data
            for mode_index in range(self.max_modes):
                strength = torch.norm(weights[:,:,:,mode_index], p='fro').cpu()
                if mode_index == 0:
                    strength_list = [strength]
                else:
                    strength_list.append(strength)
            wandb.log({layer_name+'/modes2_grad_evolution': wandb.Histogram(np.array(strength_list))})
            
        # log mode changes
        # print('modes1: {}, modes2: {}, modes3: {}'.format(self.adaptive_modes1, self.adaptive_modes2, self.adaptive_modes3))
        wandb.log({layer_name+'/modes1': self.adaptive_modes1, layer_name+'/modes2': self.adaptive_modes2, 'epoch': ep})

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 0  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # self.padding = x.shape[-1]
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        self.conv1 = FNO2d(modes, modes,  width)

    def forward(self, x):
        x = self.conv1(x)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
    
    def determine_modes(self, ep, loss):
        self.conv1.conv0.determine_modes(ep, loss, 'conv0')
        self.conv1.conv1.determine_modes(ep, loss, 'conv1')
        self.conv1.conv2.determine_modes(ep, loss, 'conv2')
        self.conv1.conv3.determine_modes(ep, loss, 'conv3')

Finetune = False

# TRAIN_PATH = '../data/darcy_s61_N1200.mat'
# TEST_PATH = '../data/darcy_s61_N1200.mat'
# TRAIN_PATH = '../data/lognormal_N1024_s61.mat'
# TEST_PATH = '../data/lognormal_N1024_s61.mat'
TRAIN_PATH = '/ngc_workspace/project/PINO/data/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = '/ngc_workspace/project/PINO/data/piececonst_r421_N1024_smooth2.mat'

ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.001
epochs = 500 #200 
step_size = 100 #50
gamma = 0.5

if Finetune:
    ntrain = 1
    ntest = 1
    batch_size = 1
    learning_rate = 0.0005
    epochs = 10000
    step_size = 1000



modes = 20
width = 64

r = 1
h = int(((421 - 1)/r) + 1)
s = h

r_data = args.r_data
s_data = int(((421 - 1)/r_data) + 1)
r_pde = args.r_pde
s_pde = int(((421 - 1)/r_pde) + 1)
print(s)


path = 'PINO_FDM_darcy_pde1_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = '../model/'+path
path_pred = '../pred/'+path+'.mat'

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]
# x_train = reader.read_field('input')[:ntrain,::r,::r][:,:s,:s]
# y_train = reader.read_field('output')[:ntrain,::r,::r][:,:s,:s]

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[-ntest:,::r,::r][:,:s,:s]
y_test = reader.read_field('sol')[-ntest:,::r,::r][:,:s,:s]
# x_test = reader.read_field('input')[-ntest:,::r,::r][:,:s,:s]
# y_test = reader.read_field('output')[-ntest:,::r,::r][:,:s,:s]

if Finetune:
    x_train = x_test
    y_train = y_test

print(torch.mean(x_train), torch.mean(y_train))

# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)
#
# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

def get_grid(s):
    grids = []
    grids.append(np.linspace(0, 1, s))
    grids.append(np.linspace(0, 1, s))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s,s,2)
    grid = torch.tensor(grid, dtype=torch.float)
    return grid
# x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
# x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)
x_train = x_train.reshape(ntrain, s, s, 1)
x_test = x_test.reshape(ntest, s, s, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

model = Net2d(args.max_modes, width).cuda()
# model = torch.load('/home/wumming/Documents/GNN-PDE/graph-pde/model/PINO_FDM_darcy_pde_N1000_ep500_m20_w64').cuda()
num_param = model.count_params()
print(num_param)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)

myloss = LpLoss(size_average=False)


def FDM_Darcy(u, a, D=1, f=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    u = u[:, 1:-1, 1:-1]
    # Du = -(ax*ux + ay*uy + a*uxx + a*uyy)

    # inner1 = torch.mean(a*(ux**2 + uy**2), dim=[1,2])
    # inner2 = torch.mean(f*u, dim=[1,2])
    # return 0.5*inner1 - inner2

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)

    return Du


def PINO_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)

    index_x = torch.cat([torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)),
                         torch.zeros(size)], dim=0).long()
    index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)), torch.zeros(size),
                         torch.tensor(range(0, size))], dim=0).long()

    boundary_u = u[:, index_x, index_y]
    truth_u = torch.zeros(boundary_u.shape, device=u.device)
    loss_bc = myloss.abs(boundary_u, truth_u)

    Du = FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    loss_f = myloss(Du, f)

    # im = (Du-f)[0].detach().cpu().numpy()
    # plt.imshow(im)
    # plt.show()

    # loss_f = FDM_Darcy(u, a)
    # loss_f = torch.mean(loss_f)
    return loss_f, loss_bc

error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()
grid = get_grid(1).cuda()
grid_data = get_grid(s_data).cuda()
grid_pde = get_grid(s_pde).cuda()

mollifier_data = torch.sin(np.pi*grid_data[...,0]) * torch.sin(np.pi*grid_data[...,1]) * 0.001
mollifier_pde = torch.sin(np.pi*grid_pde[...,0]) * torch.sin(np.pi*grid_pde[...,1]) * 0.001

print(mollifier_data.shape)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_f = 0.0
    train_l2 = 0.0
    train_loss = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()

        # data loss
        out = model(x[:,::r_data,::r_data]).reshape(batch_size, s_data, s_data)
        out = out * mollifier_data
        # out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size,-1), y[:,::r_data,::r_data].reshape(batch_size,-1))
        # a = x_normalizer.decode(x[..., 0])

        # pde loss
        a = x[:, ::r_pde, ::r_pde, 0]
        out_pde = model(x[:, ::r_pde, ::r_pde]).reshape(batch_size, s_pde, s_pde)
        out_pde = out_pde * mollifier_pde

        loss_f, loss_u = PINO_loss(out_pde, a)
        pino_loss = 1*loss_f + 1.*loss #! place to turn off data loss
        pino_loss.backward()
        # loss.backward()

        optimizer.step()
        train_l2 += loss.item()
        train_f += loss_f.item()
        train_loss += torch.tensor([loss_u, loss_f])

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    test_pino = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x[:,::r_data,::r_data]).reshape(batch_size, s_data, s_data)
            out = out * mollifier_data
            # out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y[:,::r_data,::r_data].reshape(batch_size, -1)).item()
            # a = x_normalizer.decode(x[..., 0])
            # a = x[..., 0]
            # loss_f, loss_u = PINO_loss(out, a)
            # test_pino += 1*loss_f.item() + 0*loss_u.item()

    train_l2 /= ntrain
    test_l2 /= ntest
    train_f /= ntrain
    test_pino /= len(test_loader)
    train_loss /= len(train_loader)

    # if ep % step_size == step_size-1:
    #     plt.imshow(y[0,:,:].cpu().numpy())
    #     plt.show()
    #     plt.imshow(out[0,:,:].cpu().numpy())
    #     plt.show()

    error[ep] = [train_f, train_l2, test_pino, test_l2]

    t2 = default_timer()
    print(ep, t2-t1, train_f, train_l2, test_pino, test_l2)
    # print(train_loss)
    wandb.log({'train_f': train_f, 'train_l2': train_l2, 'test_pino': test_pino, 'test_l2': test_l2, 'epoch': ep})
    
    model.determine_modes(ep, train_l2)

torch.save(model, path_model)

# pred = torch.zeros(y_test.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0
#         x, y = x.cuda(), y.cuda()
#
#         out = model(x)
#         out = y_normalizer.decode(out)
#         pred[index] = out
#
#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1
#
# scipy.io.savemat(path_pred, mdict={'pred': pred.cpu().numpy(), 'error': error})