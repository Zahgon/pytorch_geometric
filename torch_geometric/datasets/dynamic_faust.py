from itertools import product
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset


class DynamicFAUST(InMemoryDataset):

    url = 'http://dfaust.is.tue.mpg.de/'

    subjects = [
        '50002', '50004', '50007', '50009', '50020', '50021', '50022', '50025',
        '50026', '50027'
    ]
    categories = [
        'chicken_wings', 'hips', 'jiggle_on_toes', 'jumping_jacks', 'knees',
        'light_hopping_loose', 'light_hopping_stiff', 'one_leg_jump',
        'one_leg_loose', 'personal_move', 'punching', 'running_on_spot',
        'running_on_spot_bugfix', 'shake_arms', 'shake_hips', 'shake_shoulders'
    ]

    def __init__(
        self,
        root: str,
        subjects: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:

        subjects = self.subjects if subjects is None else subjects
        subjects = [sid.lower() for sid in subjects]
        for sid in subjects:
            assert sid in self.subjects
        self.subjects = subjects

        categories = self.categories if categories is None else categories
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.categories = categories

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download male registrations "
            f"'registrations_m.hdf5' and female registrations "
            f"'registrations_f.hdf5' from '{self.url}' and move it to "
            f"'{self.raw_dir}'")

    def process(self) -> None:
        import h5py

        fm = h5py.File(self.raw_paths[0], 'r')
        ff = h5py.File(self.raw_paths[1], 'r')

        face = torch.from_numpy(fm['faces'][()]).to(torch.long)
        face = face.t().contiguous()

        data_list = []
        for (sid, cat) in product(self.subjects, self.categories):
            idx = f'{sid}_{cat}'
            if idx in fm:
                pos = torch.from_numpy(fm[idx][()])
            elif idx in ff:
                pos = torch.from_numpy(ff[idx][()])
            else:
                continue
            pos = pos.permute(2, 0, 1).contiguous()
            data_list.append(Data(pos=pos, face=face, num_nodes=pos.size(1)))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
