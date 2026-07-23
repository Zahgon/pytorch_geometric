import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs

MOVIE_HEADERS = ["movieId", "title", "genres"]
USER_HEADERS = ["userId", "gender", "age", "occupation", "zipCode"]
RATING_HEADERS = ['userId', 'movieId', 'rating', 'timestamp']


class MovieLens1M(InMemoryDataset):
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'

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
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-1m')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        df = pd.read_csv(
            self.raw_paths[0],
            sep='::',
            header=None,
            index_col='movieId',
            names=MOVIE_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = df['genres'].str.get_dummies('|').values
        genres = torch.from_numpy(genres).to(torch.float)

        data['movie'].x = genres

        df = pd.read_csv(
            self.raw_paths[1],
            sep='::',
            header=None,
            index_col='userId',
            names=USER_HEADERS,
            dtype='str',
            encoding='ISO-8859-1',
            engine='python',
        )
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = df['age'].str.get_dummies().values
        age = torch.from_numpy(age).to(torch.float)

        gender = df['gender'].str.get_dummies().values
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df['occupation'].str.get_dummies().values
        occupation = torch.from_numpy(occupation).to(torch.float)

        data['user'].x = torch.cat([age, gender, occupation], dim=-1)

        df = pd.read_csv(
            self.raw_paths[2],
            sep='::',
            header=None,
            names=RATING_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_index = edge_index

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'movie'].rating = rating

        time = torch.from_numpy(df['timestamp'].values)
        data['user', 'rates', 'movie'].time = time

        data['movie', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['movie', 'rated_by', 'user'].rating = rating
        data['movie', 'rated_by', 'user'].time = time

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
