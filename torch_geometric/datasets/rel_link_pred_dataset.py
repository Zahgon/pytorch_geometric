import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class RelLinkPredDataset(InMemoryDataset):

    urls = {
        'FB15k-237': ('https://raw.githubusercontent.com/MichSchli/'
                      'RelationPrediction/master/data/FB-Toutanova')
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        assert name in ['FB15k-237']
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def num_relations(self) -> int:
        pass

    @property
    def raw_dir(self) -> str:
        pass

    @property
    def processed_dir(self) -> str:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    @property
    def raw_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        for file_name in self.raw_file_names:
            download_url(f'{self.urls[self.name]}/{file_name}', self.raw_dir)

    def process(self) -> None:
        with open(osp.join(self.raw_dir, 'entities.dict')) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: int(value) for value, key in lines}

        with open(osp.join(self.raw_dir, 'relations.dict')) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            relations_dict = {key: int(value) for value, key in lines}

        kwargs = {}
        for split in ['train', 'valid', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.txt')) as f:
                lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
                src = [entities_dict[row[0]] for row in lines]
                rel = [relations_dict[row[1]] for row in lines]
                dst = [entities_dict[row[2]] for row in lines]
                kwargs[f'{split}_edge_index'] = torch.tensor([src, dst])
                kwargs[f'{split}_edge_type'] = torch.tensor(rel)

        row, col = kwargs['train_edge_index']
        edge_type = kwargs['train_edge_type']
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_type = torch.cat([edge_type, edge_type + len(relations_dict)])

        data = Data(num_nodes=len(entities_dict), edge_index=edge_index,
                    edge_type=edge_type, **kwargs)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
