from typing import Any, Callable, List, Optional, Tuple

import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs


class EllipticBitcoinDataset(InMemoryDataset):
    url = 'https://data.pyg.org/datasets/elliptic'

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
        for file_name in self.raw_file_names:
            fs.cp(f'{self.url}/{file_name}.zip', self.raw_dir, extract=True)

    def _process_df(self, feat_df: Any, edge_df: Any,
                    class_df: Any) -> Tuple[Any, Any, Any]:
        return feat_df, edge_df, class_df

    def process(self) -> None:
        import pandas as pd

        feat_df = pd.read_csv(self.raw_paths[0], header=None)
        edge_df = pd.read_csv(self.raw_paths[1])
        class_df = pd.read_csv(self.raw_paths[2])

        columns = {0: 'txId', 1: 'time_step'}
        feat_df = feat_df.rename(columns=columns)

        feat_df, edge_df, class_df = self._process_df(
            feat_df,
            edge_df,
            class_df,
        )

        x = torch.from_numpy(feat_df.loc[:, 2:].values).to(torch.float)

        mapping = {'unknown': 2, '1': 1, '2': 0}
        class_df['class'] = class_df['class'].map(mapping)
        y = torch.from_numpy(class_df['class'].values)

        mapping = {idx: i for i, idx in enumerate(feat_df['txId'].values)}
        edge_df['txId1'] = edge_df['txId1'].map(mapping)
        edge_df['txId2'] = edge_df['txId2'].map(mapping)
        edge_index = torch.from_numpy(edge_df.values).t().contiguous()

        time_step = torch.from_numpy(feat_df['time_step'].values)
        train_mask = (time_step < 35) & (y != 2)
        test_mask = (time_step >= 35) & (y != 2)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        pass
