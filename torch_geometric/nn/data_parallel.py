import logging
import warnings
from itertools import chain

import torch

from torch_geometric.data import Batch
from torch_geometric.utils import cumsum


class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None,
                 follow_batch=None, exclude_keys=None):
        super().__init__(module, device_ids, output_device)

        warnings.warn(
            "'DataParallel' is usually much slower than "
            "'DistributedDataParallel' even on a single machine. "
            "Please consider switching to 'DistributedDataParallel' "
            "for multi-GPU training.", stacklevel=2)

        self.src_device = torch.device(f'cuda:{self.device_ids[0]}')
        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []

    def forward(self, data_list):
        """"""  # noqa: D419
        if len(data_list) == 0:
            logging.warning('DataParallel received an empty data list, which '
                            'may result in unexpected behavior.')
            return None

        if not self.device_ids or len(self.device_ids) == 1:  # Fallback
            data = Batch.from_data_list(
                data_list, follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys).to(self.src_device)
            return self.module(data)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    f"Module must have its parameters and buffers on device "
                    f"'{self.src_device}' but found one of them on device "
                    f"'{t.device}'")

        inputs = self.scatter(data_list, self.device_ids)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)

    def scatter(self, data_list, device_ids):
        num_devices = min(len(device_ids), len(data_list))

        count = torch.tensor([data.num_nodes for data in data_list])
        ptr = cumsum(count)
        device_id = num_devices * ptr.to(torch.float) / ptr[-1].item()
        device_id = (device_id[:-1] + device_id[1:]) / 2.0
        device_id = device_id.to(torch.long)  # round.
        split = cumsum(device_id.bincount())
        split = torch.unique(split, sorted=True)
        split = split.tolist()

        return [
            Batch.from_data_list(data_list[split[i]:split[i + 1]],
                                 follow_batch=self.follow_batch,
                                 exclude_keys=self.exclude_keys).to(
                                     torch.device(f'cuda:{device_ids[i]}'))
            for i in range(len(split) - 1)
        ]
