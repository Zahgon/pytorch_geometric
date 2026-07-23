from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import InMemoryDataset, TemporalData, download_url


class MyketDataset(InMemoryDataset):
    url = ('https://raw.githubusercontent.com/erfanloghmani/'
           'myket-android-application-market-dataset/main/data_int_index')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=TemporalData)

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        for file_name in self.raw_file_names:
            download_url(f'{self.url}/{file_name}', self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0], skiprows=1, header=None)

        src = torch.from_numpy(df[0].values)
        dst = torch.from_numpy(df[1].values)
        t = torch.from_numpy(df[2].values)

        x = torch.from_numpy(np.load(self.raw_paths[1])).to(torch.float)
        msg = x[dst]

        dst = dst + (int(src.max()) + 1)

        data = TemporalData(src=src, dst=dst, t=t, msg=msg)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
