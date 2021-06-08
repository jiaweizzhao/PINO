import os
import numpy as np
import torch
import torch.autograd as autograd


def get_sample(N, T, s, p, q):
    # sample p nodes from Initial Condition, p nodes from Boundary Condition, q nodes from Interior

    # sample IC
    index_ic = torch.randint(s, size=(N, p))
    sample_ic_t = torch.zeros(N, p)
    sample_ic_x = index_ic/s

    # sample BC
    sample_bc = torch.rand(size=(N, p//2))
    sample_bc_t =  torch.cat([sample_bc, sample_bc],dim=1)
    sample_bc_x = torch.cat([torch.zeros(N, p//2), torch.ones(N, p//2)],dim=1)

    # sample I
    # sample_i_t = torch.rand(size=(N,q))
    # sample_i_t = torch.rand(size=(N,q))**2
    sample_i_t = -torch.cos(torch.rand(size=(N, q))*np.pi/2) + 1
    sample_i_x = torch.rand(size=(N,q))

    sample_t = torch.cat([sample_ic_t, sample_bc_t, sample_i_t], dim=1).cuda()
    sample_t.requires_grad = True
    sample_x = torch.cat([sample_ic_x, sample_bc_x, sample_i_x], dim=1).cuda()
    sample_x.requires_grad = True
    sample = torch.stack([sample_t, sample_x], dim=-1).reshape(N, (p+p+q), 2)
    return sample, sample_t, sample_x, index_ic.long()


def get_grid(N, T, s):
    gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float).reshape(1, T, 1).repeat(N, 1, s).cuda()
    gridt.requires_grad = True
    gridx = torch.tensor(np.linspace(0, 1, s+1)[:-1], dtype=torch.float).reshape(1, 1, s).repeat(N, T, 1).cuda()
    gridx.requires_grad = True
    grid = torch.stack([gridt, gridx], dim=-1).reshape(N, T*s, 2)
    return grid, gridt, gridx


def get_grid3d(S, T):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def set_grad(tensors, flag=True):
    for p in tensors:
        p.requires_grad = flag


def zero_grad(params):
    '''
    set grad field to 0
    '''
    if isinstance(params, torch.Tensor):
        if params.grad is not None:
            params.grad.zero_()
    else:
        for p in params:
            if p.grad is not None:
                p.grad.zero_()


def count_params(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count


def save_checkpoint(path, name, model, optimizer=None):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0

    torch.save({
        'model': model_state_dict,
        'optim': optim_dict
    }, ckpt_dir + name)
    print('Checkpoint is saved at %s' % ckpt_dir + name)