import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act


def relu():
    return torch.nn.ReLU(inplace=cfg.mem.inplace)


def selu():
    return torch.nn.SELU(inplace=cfg.mem.inplace)


def prelu():
    pass


def elu():
    return torch.nn.ELU(inplace=cfg.mem.inplace)


def lrelu_01():
    pass


def lrelu_025():
    pass


def lrelu_05():
    pass


if cfg is not None:
    register_act('relu', relu)
    register_act('selu', selu)
    register_act('prelu', prelu)
    register_act('elu', elu)
    register_act('lrelu_01', lrelu_01)
    register_act('lrelu_025', lrelu_025)
    register_act('lrelu_05', lrelu_05)
