import json
import os
import os.path as osp
from typing import Callable, Dict, List, Literal, Optional

import torch
import tqdm
from torch import Tensor

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_tar,
)


class OPFDataset(InMemoryDataset):
    url = 'https://storage.googleapis.com/gridopt-dataset'

    def __init__(
        self,
        root: str,
        split: Literal['train', 'val', 'test'] = 'train',
        case_name: Literal[
            'pglib_opf_case14_ieee',
            'pglib_opf_case30_ieee',
            'pglib_opf_case57_ieee',
            'pglib_opf_case118_ieee',
            'pglib_opf_case500_goc',
            'pglib_opf_case2000_goc',
            'pglib_opf_case6470_rte',
            'pglib_opf_case4661_sdet'
            'pglib_opf_case10000_goc',
            'pglib_opf_case13659_pegase',
        ] = 'pglib_opf_case14_ieee',
        num_groups: int = 20,
        topological_perturbations: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:

        self.split = split
        self.case_name = case_name
        self.num_groups = num_groups
        self.topological_perturbations = topological_perturbations

        self._release = 'dataset_release_1'
        if topological_perturbations:
            self._release += '_nminusone'

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        idx = self.processed_file_names.index(f'{split}.pt')
        self.load(self.processed_paths[idx])

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
    def processed_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        for name in self.raw_file_names:
            url = f'{self.url}/{self._release}/{name}'
            path = download_url(url, self.raw_dir)
            extract_tar(path, self.raw_dir)

    def process(self) -> None:
        train_data_list = []
        val_data_list = []
        test_data_list = []

        for group in tqdm.tqdm(range(self.num_groups)):
            tmp_dir = osp.join(
                self.raw_dir,
                'gridopt-dataset-tmp',
                self._release,
                self.case_name,
                f'group_{group}',
            )

            for name in os.listdir(tmp_dir):
                with open(osp.join(tmp_dir, name)) as f:
                    obj = json.load(f)

                grid = obj['grid']
                solution = obj['solution']
                metadata = obj['metadata']

                data = HeteroData()
                data.x = torch.tensor(grid['context']).view(-1)

                data.objective = torch.tensor(metadata['objective'])

                data['bus'].x = torch.tensor(grid['nodes']['bus'])
                data['bus'].y = torch.tensor(solution['nodes']['bus'])

                data['generator'].x = torch.tensor(grid['nodes']['generator'])
                data['generator'].y = torch.tensor(
                    solution['nodes']['generator'])

                data['load'].x = torch.tensor(grid['nodes']['load'])

                data['shunt'].x = torch.tensor(grid['nodes']['shunt'])

                data['bus', 'ac_line', 'bus'].edge_index = (  #
                    extract_edge_index(obj, 'ac_line'))
                data['bus', 'ac_line', 'bus'].edge_attr = torch.tensor(
                    grid['edges']['ac_line']['features'])
                data['bus', 'ac_line', 'bus'].edge_label = torch.tensor(
                    solution['edges']['ac_line']['features'])

                data['bus', 'transformer', 'bus'].edge_index = (  #
                    extract_edge_index(obj, 'transformer'))
                data['bus', 'transformer', 'bus'].edge_attr = torch.tensor(
                    grid['edges']['transformer']['features'])
                data['bus', 'transformer', 'bus'].edge_label = torch.tensor(
                    solution['edges']['transformer']['features'])

                data['generator', 'generator_link', 'bus'].edge_index = (  #
                    extract_edge_index(obj, 'generator_link'))
                data['bus', 'generator_link', 'generator'].edge_index = (  #
                    extract_edge_index_rev(obj, 'generator_link'))

                data['load', 'load_link', 'bus'].edge_index = (  #
                    extract_edge_index(obj, 'load_link'))
                data['bus', 'load_link', 'load'].edge_index = (  #
                    extract_edge_index_rev(obj, 'load_link'))

                data['shunt', 'shunt_link', 'bus'].edge_index = (  #
                    extract_edge_index(obj, 'shunt_link'))
                data['bus', 'shunt_link', 'shunt'].edge_index = (  #
                    extract_edge_index_rev(obj, 'shunt_link'))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                i = int(name.split('.')[0].split('_')[1])
                train_limit = int(15_000 * self.num_groups * 0.9)
                val_limit = train_limit + int(15_000 * self.num_groups * 0.05)
                if i < train_limit:
                    train_data_list.append(data)
                elif i < val_limit:
                    val_data_list.append(data)
                else:
                    test_data_list.append(data)

        self.save(train_data_list, self.processed_paths[0])
        self.save(val_data_list, self.processed_paths[1])
        self.save(test_data_list, self.processed_paths[2])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'split={self.split}, '
                f'case_name={self.case_name}, '
                f'topological_perturbations={self.topological_perturbations})')


def extract_edge_index(obj: Dict, edge_name: str) -> Tensor:
    return torch.tensor([
        obj['grid']['edges'][edge_name]['senders'],
        obj['grid']['edges'][edge_name]['receivers'],
    ])


def extract_edge_index_rev(obj: Dict, edge_name: str) -> Tensor:
    return torch.tensor([
        obj['grid']['edges'][edge_name]['receivers'],
        obj['grid']['edges'][edge_name]['senders'],
    ])
