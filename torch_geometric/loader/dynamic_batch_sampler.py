from typing import Iterator, List, Optional

import torch

from torch_geometric.data import Dataset


class DynamicBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self,
        dataset: Dataset,
        max_num: int,
        mode: str = 'node',
        shuffle: bool = False,
        skip_too_big: bool = False,
        num_steps: Optional[int] = None,
    ):
        if max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value "
                             f"(got {max_num})")
        if mode not in ['node', 'edge']:
            raise ValueError(f"`mode` choice should be either "
                             f"'node' or 'edge' (got '{mode}')")

        self.dataset = dataset
        self.max_num = max_num
        self.mode = mode
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.num_steps = num_steps
        self.max_steps = num_steps or len(dataset)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = range(len(self.dataset))

        samples: List[int] = []
        current_num: int = 0
        num_steps: int = 0
        num_processed: int = 0

        while (num_processed < len(self.dataset)
               and num_steps < self.max_steps):

            for i in indices[num_processed:]:
                data = self.dataset[i]
                num = data.num_nodes if self.mode == 'node' else data.num_edges

                if current_num + num > self.max_num:
                    if current_num == 0:
                        if self.skip_too_big:
                            continue
                    else:  # Mini-batch filled:
                        break

                samples.append(i)
                num_processed += 1
                current_num += num

            yield samples
            samples: List[int] = []
            current_num = 0
            num_steps += 1

    def __len__(self) -> int:
        if self.num_steps is None:
            raise ValueError(f"The length of '{self.__class__.__name__}' is "
                             f"undefined since the number of steps per epoch "
                             f"is ambiguous. Either specify `num_steps` or "
                             f"use a static batch sampler.")

        return self.num_steps
