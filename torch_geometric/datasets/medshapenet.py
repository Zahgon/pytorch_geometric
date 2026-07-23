import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset


class MedShapeNet(InMemoryDataset):
    def __init__(
        self,
        root: str,
        size: int = 100,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.size = size
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        path = self.processed_paths[0]
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    @property
    def raw_paths(self) -> List[str]:
        pass

    def process(self) -> None:
        import urllib3
        from MedShapeNet import MedShapeNet as msn

        msn_instance = msn(timeout=120)

        urllib3.HTTPConnectionPool("medshapenet.ddns.net", maxsize=50)

        list_of_datasets = msn_instance.datasets(False)
        list_of_datasets = list(
            filter(
                lambda x: x not in [
                    'medshapenetcore/ASOCA', 'medshapenetcore/AVT',
                    'medshapenetcore/AutoImplantCraniotomy',
                    'medshapenetcore/FaceVR'
                ], list_of_datasets))

        subset = []
        for dataset in list_of_datasets:
            parts = dataset.split("/")
            self.newpath = self.root + '/' + parts[1 if len(parts) > 1 else 0]
            if not os.path.exists(self.newpath):
                os.makedirs(self.newpath)
            stl_files = msn_instance.dataset_files(dataset, '.stl')
            subset.extend(stl_files[:self.size])

            for stl_file in stl_files[:self.size]:
                msn_instance.download_stl_as_numpy(bucket_name=dataset,
                                                   stl_file=stl_file,
                                                   output_dir=self.newpath,
                                                   print_output=False)

        class_mapping = {
            '3DTeethSeg': 0,
            'CoronaryArteries': 1,
            'FLARE': 2,
            'KITS': 3,
            'PULMONARY': 4,
            'SurgicalInstruments': 5,
            'ThoracicAorta_Saitta': 6,
            'ToothFairy': 7
        }

        for dataset, path in zip([subset], self.processed_paths):
            data_list = []
            for item in dataset:
                class_name = item.split("/")[0]
                item = item.split("stl")[0]
                target = class_mapping[class_name]
                file = osp.join(self.root, item + 'npz')

                data = np.load(file)
                pre_data_list = Data(
                    pos=torch.tensor(data["vertices"], dtype=torch.float),
                    face=torch.tensor(data["faces"],
                                      dtype=torch.long).t().contiguous())
                pre_data_list.y = torch.tensor([target], dtype=torch.long)
                data_list.append(pre_data_list)

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, path)
