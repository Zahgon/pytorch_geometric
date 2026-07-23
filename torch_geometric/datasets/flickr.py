import json
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_google_url


class Flickr(InMemoryDataset):
    adj_full_id = '1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy'
    feats_id = '1join-XdvX3anJU_MLVtick7MgeAQiWIZ'
    class_map_id = '1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9'
    role_id = '1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        download_google_url(self.adj_full_id, self.raw_dir, 'adj_full.npz')
        download_google_url(self.feats_id, self.raw_dir, 'feats.npy')
        download_google_url(self.class_map_id, self.raw_dir, 'class_map.json')
        download_google_url(self.role_id, self.raw_dir, 'role.json')

    def process(self) -> None:
        import scipy.sparse as sp

        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)

        self.save([data], self.processed_paths[0])
