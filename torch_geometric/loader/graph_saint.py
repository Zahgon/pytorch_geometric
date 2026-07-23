import os.path as osp
from typing import Optional

import torch
from tqdm import tqdm

from torch_geometric.io import fs
from torch_geometric.typing import SparseTensor


class GraphSAINTSampler(torch.utils.data.DataLoader):
    def __init__(self, data, batch_size: int, num_steps: int = 1,
                 sample_coverage: int = 0, save_dir: Optional[str] = None,
                 log: bool = True, **kwargs):

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        assert data.edge_index is not None
        assert 'node_norm' not in data
        assert 'edge_norm' not in data
        assert not data.edge_index.is_cuda

        self.num_steps = num_steps
        self._batch_size = batch_size
        self.sample_coverage = sample_coverage
        self.save_dir = save_dir
        self.log = log

        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            value=torch.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(N, N))

        self.data = data

        super().__init__(self, batch_size=1, collate_fn=self._collate,
                         **kwargs)

        if self.sample_coverage > 0:
            path = osp.join(save_dir or '', self._filename)
            if save_dir is not None and osp.exists(path):  # pragma: no cover
                self.node_norm, self.edge_norm = fs.torch_load(path)
            else:
                self.node_norm, self.edge_norm = self._compute_norm()
                if save_dir is not None:  # pragma: no cover
                    torch.save((self.node_norm, self.edge_norm), path)

    @property
    def _filename(self):
        pass

    def __len__(self):
        return self.num_steps

    def _sample_nodes(self, batch_size):
        raise NotImplementedError

    def __getitem__(self, idx):
        node_idx = self._sample_nodes(self._batch_size).unique()
        adj, _ = self.adj.saint_subgraph(node_idx)
        return node_idx, adj

    def _collate(self, data_list):
        assert len(data_list) == 1
        node_idx, adj = data_list[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in self.data:
            if key in ['edge_index', 'num_nodes']:
                continue
            if isinstance(item, torch.Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            elif isinstance(item, torch.Tensor) and item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        if self.sample_coverage > 0:
            data.node_norm = self.node_norm[node_idx]
            data.edge_norm = self.edge_norm[edge_idx]

        return data

    def _compute_norm(self):
        node_count = torch.zeros(self.N, dtype=torch.float)
        edge_count = torch.zeros(self.E, dtype=torch.float)

        loader = torch.utils.data.DataLoader(self, batch_size=200,
                                             collate_fn=lambda x: x,
                                             num_workers=self.num_workers)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.N * self.sample_coverage)
            pbar.set_description('Compute GraphSAINT normalization')

        num_samples = total_sampled_nodes = 0
        while total_sampled_nodes < self.N * self.sample_coverage:
            for data in loader:
                for node_idx, adj in data:
                    edge_idx = adj.storage.value()
                    node_count[node_idx] += 1
                    edge_count[edge_idx] += 1
                    total_sampled_nodes += node_idx.size(0)

                    if self.log:  # pragma: no cover
                        pbar.update(node_idx.size(0))
            num_samples += self.num_steps

        if self.log:  # pragma: no cover
            pbar.close()

        row, _, edge_idx = self.adj.coo()
        t = torch.empty_like(edge_count).scatter_(0, edge_idx, node_count[row])
        edge_norm = (t / edge_count).clamp_(0, 1e4)
        edge_norm[torch.isnan(edge_norm)] = 0.1

        node_count[node_count == 0] = 0.1
        node_norm = num_samples / node_count / self.N

        return node_norm, edge_norm


class GraphSAINTNodeSampler(GraphSAINTSampler):
    def _sample_nodes(self, batch_size):
        pass


class GraphSAINTEdgeSampler(GraphSAINTSampler):
    def _sample_nodes(self, batch_size):
        pass


class GraphSAINTRandomWalkSampler(GraphSAINTSampler):
    def __init__(self, data, batch_size: int, walk_length: int,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir: Optional[str] = None, log: bool = True, **kwargs):
        self.walk_length = walk_length
        super().__init__(data, batch_size, num_steps, sample_coverage,
                         save_dir, log, **kwargs)

    @property
    def _filename(self):
        pass

    def _sample_nodes(self, batch_size):
        pass
