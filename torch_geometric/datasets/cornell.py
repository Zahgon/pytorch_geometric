import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data.hypergraph_data import HyperGraphData


class CornellTemporalHyperGraphDataset(InMemoryDataset):
    names = [
        'email-Eu',
        'email-Enron',
        'NDC-classes',
        'tags-math-sx',
        'email-Eu-25',
        'NDC-substances',
        'congress-bills',
        'tags-ask-ubuntu',
        'email-Enron-25',
        'NDC-classes-25',
        'threads-ask-ubuntu',
        'contact-high-school',
        'NDC-substances-25',
        'congress-bills-25',
        'contact-primary-school',
    ]
    url = ('https://huggingface.co/datasets/SauravMaheshkar/{}/raw/main/'
           'processed/{}/{}')

    def __init__(
        self,
        root: str,
        name: str,
        split: str = 'train',
        setting: str = 'transductive',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert name in self.names
        assert setting in ['transductive', 'inductive']

        self.name = name
        self.setting = setting

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"Split '{split}' not found")

        self.load(path)

    @property
    def raw_dir(self) -> str:
        pass

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_dir(self) -> str:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        for filename in self.raw_file_names:
            url = self.url.format(self.name, self.setting, filename)
            download_url(url, self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            df = pd.read_csv(raw_path)

            data_list = []
            for i, row in df.iterrows():
                edge_indices: List[List[int]] = [[], []]
                for node in eval(row['nodes']):  # str(list) -> list:
                    edge_indices[0].append(node)
                    edge_indices[1].append(i)  # Use `i` as hyper-edge index.

                x = torch.tensor([[row['timestamp']]], dtype=torch.float)
                edge_index = torch.tensor(edge_indices)

                data = HyperGraphData(x=x, edge_index=edge_index)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            self.save(data_list, path)
