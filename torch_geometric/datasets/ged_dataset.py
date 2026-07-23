import glob
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_google_url,
    extract_tar,
    extract_zip,
)
from torch_geometric.io import fs
from torch_geometric.utils import one_hot, to_undirected


class GEDDataset(InMemoryDataset):
    datasets = {
        'AIDS700nef': {
            'id': '10czBPJDEzEDI2tq7Z7mkBjLhj55F-a2z',
            'extract': extract_zip,
            'pickle': '1OpV4bCHjBkdpqI6H5Mg0-BqlA2ee2eBW',
        },
        'LINUX': {
            'id': '1nw0RRVgyLpit4V4XFQyDy0pI6wUEXSOI',
            'extract': extract_tar,
            'pickle': '14FDm3NSnrBvB7eNpLeGy5Bz6FjuCSF5v',
        },
        'ALKANE': {
            'id': '1-LmxaWW3KulLh00YqscVEflbqr0g4cXt',
            'extract': extract_tar,
            'pickle': '15BpvMuHx77-yUGYgM27_sQett02HQNYu',
        },
        'IMDBMulti': {
            'id': '12QxZ7EhYA7pJiF4cO-HuE8szhSOWcfST',
            'extract': extract_zip,
            'pickle': '1wy9VbZvZodkixxVIOuRllC-Lp-0zdoYZ',
        },
    }

    types = [
        'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
        'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
        'Sb', 'Se', 'Ni', 'Te'
    ]

    def __init__(
        self,
        root: str,
        name: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_ged.pt')
        self.ged = fs.torch_load(path)
        path = osp.join(self.processed_dir, f'{self.name}_norm_ged.pt')
        self.norm_ged = fs.torch_load(path)

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        id = self.datasets[self.name]['id']
        assert isinstance(id, str)
        path = download_google_url(id, self.raw_dir, 'data')
        extract_fn = self.datasets[self.name]['extract']
        assert callable(extract_fn)
        extract_fn(path, self.raw_dir)
        os.unlink(path)

        id = self.datasets[self.name]['pickle']
        assert isinstance(id, str)
        path = download_google_url(id, self.raw_dir, 'ged.pickle')

    def process(self) -> None:
        import networkx as nx

        ids, Ns = [], []
        for r_path, p_path in zip(self.raw_paths, self.processed_paths):
            names = glob.glob(osp.join(r_path, '*.gexf'))
            ids.append(sorted([int(osp.basename(i)[:-5]) for i in names]))

            data_list = []
            for i, idx in enumerate(ids[-1]):
                i = i if len(ids) == 1 else i + len(ids[0])
                G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))
                mapping = {name: j for j, name in enumerate(G.nodes())}
                G = nx.relabel_nodes(G, mapping)
                Ns.append(G.number_of_nodes())

                edge_index = torch.tensor(list(G.edges)).t().contiguous()
                if edge_index.numel() == 0:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_index = to_undirected(edge_index, num_nodes=Ns[-1])

                data = Data(edge_index=edge_index, i=i)
                data.num_nodes = Ns[-1]

                if self.name == 'AIDS700nef':
                    assert data.num_nodes is not None
                    x = torch.zeros(data.num_nodes, dtype=torch.long)
                    for node, info in G.nodes(data=True):
                        x[int(node)] = self.types.index(info['type'])
                    data.x = one_hot(x, num_classes=len(self.types))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            self.save(data_list, p_path)

        assoc = {idx: i for i, idx in enumerate(ids[0])}
        assoc.update({idx: i + len(ids[0]) for i, idx in enumerate(ids[1])})

        path = osp.join(self.raw_dir, self.name, 'ged.pickle')
        mat = torch.full((len(assoc), len(assoc)), float('inf'))
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            xs, ys, gs = [], [], []
            for (_x, _y), g in obj.items():
                xs += [assoc[_x]]
                ys += [assoc[_y]]
                gs += [g]
            x, y = torch.tensor(xs), torch.tensor(ys)
            ged = torch.tensor(gs, dtype=torch.float)
            mat[x, y], mat[y, x] = ged, ged

        path = osp.join(self.processed_dir, f'{self.name}_ged.pt')
        torch.save(mat, path)

        N = torch.tensor(Ns, dtype=torch.float)
        norm_mat = mat / (0.5 * (N.view(-1, 1) + N.view(1, -1)))

        path = osp.join(self.processed_dir, f'{self.name}_norm_ged.pt')
        torch.save(norm_mat, path)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
