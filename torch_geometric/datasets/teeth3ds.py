import json
import os
import os.path as osp
from glob import glob
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class Teeth3DS(InMemoryDataset):
    urls = {
        'data_part_1.zip':
        'https://osf.io/download/qhprs/',
        'data_part_2.zip':
        'https://osf.io/download/4pwnr/',
        'data_part_3.zip':
        'https://osf.io/download/frwdp/',
        'data_part_4.zip':
        'https://osf.io/download/2arn4/',
        'data_part_5.zip':
        'https://osf.io/download/xrz5f/',
        'data_part_6.zip':
        'https://osf.io/download/23hgq/',
        'data_part_7.zip':
        'https://osf.io/download/u83ad/',
        'train_test_split':
        'https://files.de-1.osf.io/v1/'
        'resources/xctdy/providers/osfstorage/?zip='
    }

    sample_url = {
        'teeth3ds_sample': 'https://osf.io/download/vr38s/',
    }

    landmarks_urls = {
        '3DTeethLand_landmarks_train.zip': 'https://osf.io/download/k5hbj/',
        '3DTeethLand_landmarks_test.zip': 'https://osf.io/download/sqw5e/',
    }

    def __init__(
        self,
        root: str,
        split:
        str = 'Teeth3DS',  # [3DTeethSeg22_challenge, 3DTeethLand_challenge]
        train: bool = True,
        num_samples: int = 30000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:

        self.mode = 'training' if train else 'testing'
        self.split = split
        self.num_samples = num_samples

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)

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
        if self.split == 'sample':
            for key, url in self.sample_url.items():
                path = download_url(url, self.root, filename=key)
                extract_zip(path, self.raw_dir)
                os.unlink(path)
        else:
            for key, url in self.urls.items():
                path = download_url(url, self.root, filename=key)
                extract_zip(path, self.raw_dir)
                os.unlink(path)
            for key, url in self.landmarks_urls.items():
                path = download_url(url, self.root, filename=key)
                extract_zip(path, self.raw_dir)  # Extract each downloaded part
                os.unlink(path)

    def process_file(self, file_path: str) -> Optional[Data]:
        """Processes the input file path to load mesh data, annotations,
        and prepare the input features for a graph-based deep learning model.
        """
        import trimesh
        from fpsample import bucket_fps_kdline_sampling

        mesh = trimesh.load_mesh(file_path)

        if isinstance(mesh, list):
            mesh = mesh[0]

        vertices = mesh.vertices
        vertex_normals = mesh.vertex_normals

        if len(vertices) < self.num_samples:
            sampled_indices = np.random.choice(
                len(vertices),
                self.num_samples,
                replace=True,
            )
        else:
            sampled_indices = bucket_fps_kdline_sampling(
                vertices,
                self.num_samples,
                h=5,
                start_idx=0,
            )

        if len(sampled_indices) != self.num_samples:
            raise RuntimeError(f"Sampled points mismatch, expected "
                               f"{self.num_samples} points, but got "
                               f"{len(sampled_indices)} for '{file_path}'")

        pos = torch.tensor(vertices[sampled_indices], dtype=torch.float)
        x = torch.tensor(vertex_normals[sampled_indices], dtype=torch.float)

        seg_annotation_path = file_path.replace('.obj', '.json')
        if osp.exists(seg_annotation_path):
            with open(seg_annotation_path) as f:
                seg_annotations = json.load(f)
            y = torch.tensor(
                np.asarray(seg_annotations['labels'])[sampled_indices],
                dtype=torch.float)
            instances = torch.tensor(
                np.asarray(seg_annotations['instances'])[sampled_indices],
                dtype=torch.float)
        else:
            y = torch.empty(0, 3)
            instances = torch.empty(0, 3)

        landmarks_annotation_path = file_path.replace('.obj', '__kpt.json')

        keypoints_dict: Dict[str, List] = {
            key: []
            for key in [
                'Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint',
                'FacialPoint'
            ]
        }
        keypoint_tensors: Dict[str, torch.Tensor] = {
            key: torch.empty(0, 3)
            for key in [
                'Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint',
                'FacialPoint'
            ]
        }
        if osp.exists(landmarks_annotation_path):
            with open(landmarks_annotation_path) as f:
                landmarks_annotations = json.load(f)

            for keypoint in landmarks_annotations['objects']:
                keypoints_dict[keypoint['class']].extend(keypoint['coord'])

            keypoint_tensors = {
                k: torch.tensor(np.asarray(v),
                                dtype=torch.float).reshape(-1, 3)
                for k, v in keypoints_dict.items()
            }

        data = Data(
            pos=pos,
            x=x,
            y=y,
            instances=instances,
            jaw=file_path.split('.obj')[0].split('_')[1],
            mesial=keypoint_tensors['Mesial'],
            distal=keypoint_tensors['Distal'],
            cusp=keypoint_tensors['Cusp'],
            inner_point=keypoint_tensors['InnerPoint'],
            outer_point=keypoint_tensors['OuterPoint'],
            facial_point=keypoint_tensors['FacialPoint'],
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data

    def process(self) -> None:
        for file in tqdm(self.processed_file_names):
            name = file.split('.')[0]
            path = osp.join(self.raw_dir, '**', '*', name + '.obj')
            paths = glob(path)
            if len(paths) == 1:
                data = self.process_file(paths[0])
                torch.save(data, osp.join(self.processed_dir, file))

    def len(self) -> int:
        pass

    def get(self, idx: int) -> Data:
        return torch.load(
            osp.join(self.processed_dir, self.processed_file_names[idx]),
            weights_only=False,
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'mode={self.mode}, split={self.split})')
