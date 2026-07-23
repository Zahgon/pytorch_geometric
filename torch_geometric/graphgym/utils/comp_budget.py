import math

from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model


def params_count(model):
    """Computes the number of parameters.

    Args:
        model (nn.Module): PyTorch model
    """
    return sum([p.numel() for p in model.parameters()])


def get_stats():
    pass


def match_computation(stats_baseline, key=None, mode='sqrt'):
    pass


def dict_to_stats(cfg_dict):
    pass


def match_baseline_cfg(cfg_dict, cfg_dict_baseline, verbose=True):
    pass
