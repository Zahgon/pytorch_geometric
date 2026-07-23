import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class LastFM(InMemoryDataset):
    url = 'https://www.dropbox.com/s/jvlbs09pz6zwcka/LastFM_processed.zip?dl=1'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self) -> None:
        import scipy.sparse as sp

        data = HeteroData()

        node_type_idx = np.load(osp.join(self.raw_dir, 'node_types.npy'))
        node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)

        node_types = ['user', 'artist', 'tag']
        for i, node_type in enumerate(node_types):
            data[node_type].num_nodes = int((node_type_idx == i).sum())

        pos_split = np.load(
            osp.join(self.raw_dir, 'train_val_test_pos_user_artist.npz'))
        neg_split = np.load(
            osp.join(self.raw_dir, 'train_val_test_neg_user_artist.npz'))

        for name in ['train', 'val', 'test']:
            if name != 'train':
                edge_index = pos_split[f'{name}_pos_user_artist']
                edge_index = torch.from_numpy(edge_index)
                edge_index = edge_index.t().to(torch.long).contiguous()
                data['user', 'artist'][f'{name}_pos_edge_index'] = edge_index

            edge_index = neg_split[f'{name}_neg_user_artist']
            edge_index = torch.from_numpy(edge_index)
            edge_index = edge_index.t().to(torch.long).contiguous()
            data['user', 'artist'][f'{name}_neg_edge_index'] = edge_index

        s = {}
        N_u = data['user'].num_nodes
        N_a = data['artist'].num_nodes
        N_t = data['tag'].num_nodes
        s['user'] = (0, N_u)
        s['artist'] = (N_u, N_u + N_a)
        s['tag'] = (N_u + N_a, N_u + N_a + N_t)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).to(torch.long)
                col = torch.from_numpy(A_sub.col).to(torch.long)
                data[src, dst].edge_index = torch.stack([row, col], dim=0)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
