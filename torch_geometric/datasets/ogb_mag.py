import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class OGB_MAG(InMemoryDataset):

    url = 'http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip'
    urls = {
        'metapath2vec': ('https://data.pyg.org/datasets/'
                         'mag_metapath2vec_emb.zip'),
        'transe': ('https://data.pyg.org/datasets/'
                   'mag_transe_emb.zip'),
    }

    def __init__(
        self,
        root: str,
        preprocess: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        assert self.preprocess in [None, 'metapath2vec', 'transe']
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def num_classes(self) -> int:
        pass

    @property
    def raw_dir(self) -> str:
        pass

    @property
    def processed_dir(self) -> str:
        pass

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        if not all([osp.exists(f) for f in self.raw_paths[:5]]):
            path = download_url(self.url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            for file_name in ['node-feat', 'node-label', 'relations']:
                path = osp.join(self.raw_dir, 'mag', 'raw', file_name)
                shutil.move(path, self.raw_dir)
            path = osp.join(self.raw_dir, 'mag', 'split')
            shutil.move(path, self.raw_dir)
            path = osp.join(self.raw_dir, 'mag', 'raw', 'num-node-dict.csv.gz')
            shutil.move(path, self.raw_dir)
            fs.rm(osp.join(self.raw_dir, 'mag'))
            os.remove(osp.join(self.raw_dir, 'mag.zip'))
        if self.preprocess is not None:
            path = download_url(self.urls[self.preprocess], self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.remove(path)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        path = osp.join(self.raw_dir, 'node-feat', 'paper', 'node-feat.csv.gz')
        x_paper = pd.read_csv(path, compression='gzip', header=None,
                              dtype=np.float32).values
        data['paper'].x = torch.from_numpy(x_paper)

        path = osp.join(self.raw_dir, 'node-feat', 'paper', 'node_year.csv.gz')
        year_paper = pd.read_csv(path, compression='gzip', header=None,
                                 dtype=np.int64).values
        data['paper'].year = torch.from_numpy(year_paper).view(-1)

        path = osp.join(self.raw_dir, 'node-label', 'paper',
                        'node-label.csv.gz')
        y_paper = pd.read_csv(path, compression='gzip', header=None,
                              dtype=np.int64).values.flatten()
        data['paper'].y = torch.from_numpy(y_paper)

        if self.preprocess is None:
            path = osp.join(self.raw_dir, 'num-node-dict.csv.gz')
            num_nodes_df = pd.read_csv(path, compression='gzip')
            for node_type in ['author', 'institution', 'field_of_study']:
                data[node_type].num_nodes = num_nodes_df[node_type].tolist()[0]
        else:
            emb_dict = fs.torch_load(self.raw_paths[-1])
            for key, value in emb_dict.items():
                if key != 'paper':
                    data[key].x = value

        for edge_type in [('author', 'affiliated_with', 'institution'),
                          ('author', 'writes', 'paper'),
                          ('paper', 'cites', 'paper'),
                          ('paper', 'has_topic', 'field_of_study')]:

            f = '___'.join(edge_type)
            path = osp.join(self.raw_dir, 'relations', f, 'edge.csv.gz')
            edge_index = pd.read_csv(path, compression='gzip', header=None,
                                     dtype=np.int64).values
            edge_index = torch.from_numpy(edge_index).t().contiguous()
            data[edge_type].edge_index = edge_index

        for f, v in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            path = osp.join(self.raw_dir, 'split', 'time', 'paper',
                            f'{f}.csv.gz')
            idx = pd.read_csv(path, compression='gzip', header=None,
                              dtype=np.int64).values.flatten()
            idx = torch.from_numpy(idx)
            mask = torch.zeros(data['paper'].num_nodes, dtype=torch.bool)
            mask[idx] = True
            data['paper'][f'{v}_mask'] = mask

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return 'ogbn-mag()'
