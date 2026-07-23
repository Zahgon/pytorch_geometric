import os
import os.path as osp
from typing import Callable, List, Optional

from torch_geometric.data import InMemoryDataset, download_url, extract_tar
from torch_geometric.io import fs, read_planetoid_data


class NELL(InMemoryDataset):

    url = 'http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz'

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
        path = download_url(self.url, self.root)
        extract_tar(path, self.root)
        os.unlink(path)
        fs.rm(self.raw_dir)
        os.rename(osp.join(self.root, 'nell_data'), self.raw_dir)

    def process(self) -> None:
        data = read_planetoid_data(self.raw_dir, 'nell.0.001')
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
