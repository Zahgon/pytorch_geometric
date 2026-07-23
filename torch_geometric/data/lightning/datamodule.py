import copy
import inspect
import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch

from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.loader import DataLoader, LinkLoader, NodeLoader
from torch_geometric.sampler import BaseSampler, NeighborSampler
from torch_geometric.typing import InputEdges, InputNodes, OptTensor

try:
    from lightning.pytorch import LightningDataModule as _LightningDataModule
    _pl_is_available = True
except ImportError:
    try:
        from pytorch_lightning import \
            LightningDataModule as _LightningDataModule
        _pl_is_available = True
    except ImportError:
        _pl_is_available = False
        _LightningDataModule = object


class LightningDataModule(_LightningDataModule):
    def __init__(self, has_val: bool, has_test: bool, **kwargs: Any) -> None:
        super().__init__()

        if not _pl_is_available:
            raise ModuleNotFoundError(
                "No module named 'pytorch_lightning' (or 'lightning') found "
                "in your Python environment. Run 'pip install "
                "pytorch_lightning' or 'pip install lightning'")

        if not has_val:
            self.val_dataloader = None  # type: ignore

        if not has_test:
            self.test_dataloader = None  # type: ignore

        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('num_workers', 0)
        kwargs.setdefault('pin_memory', True)
        kwargs.setdefault('persistent_workers',
                          kwargs.get('num_workers', 0) > 0)

        if 'shuffle' in kwargs:
            warnings.warn(
                f"The 'shuffle={kwargs['shuffle']}' option is "
                f"ignored in '{self.__class__.__name__}'. Remove it "
                f"from the argument list to disable this warning",
                stacklevel=2)
            del kwargs['shuffle']

        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({kwargs_repr(**self.kwargs)})'


class LightningData(LightningDataModule):
    def __init__(
        self,
        data: Union[Data, HeteroData],
        has_val: bool,
        has_test: bool,
        loader: str = 'neighbor',
        graph_sampler: Optional[BaseSampler] = None,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('num_workers', 0)

        if graph_sampler is not None:
            loader = 'custom'

        if loader not in ['full', 'neighbor', 'link_neighbor', 'custom']:
            raise ValueError(f"Undefined 'loader' option (got '{loader}')")

        if loader == 'full' and kwargs['batch_size'] != 1:
            warnings.warn(
                f"Re-setting 'batch_size' to 1 in "
                f"'{self.__class__.__name__}' for loader='full' "
                f"(got '{kwargs['batch_size']}')", stacklevel=2)
            kwargs['batch_size'] = 1

        if loader == 'full' and kwargs['num_workers'] != 0:
            warnings.warn(
                f"Re-setting 'num_workers' to 0 in "
                f"'{self.__class__.__name__}' for loader='full' "
                f"(got '{kwargs['num_workers']}')", stacklevel=2)
            kwargs['num_workers'] = 0

        if loader == 'full' and kwargs.get('sampler') is not None:
            warnings.warn(
                "'sampler' option is not supported for "
                "loader='full'", stacklevel=2)
            kwargs.pop('sampler', None)

        if loader == 'full' and kwargs.get('batch_sampler') is not None:
            warnings.warn(
                "'batch_sampler' option is not supported for "
                "loader='full'", stacklevel=2)
            kwargs.pop('batch_sampler', None)

        super().__init__(has_val, has_test, **kwargs)

        if loader == 'full':
            if kwargs.get('pin_memory', False):
                warnings.warn(
                    f"Re-setting 'pin_memory' to 'False' in "
                    f"'{self.__class__.__name__}' for loader='full' "
                    f"(got 'True')", stacklevel=2)
            self.kwargs['pin_memory'] = False

        self.data = data
        self.loader = loader


        if loader in ['neighbor', 'link_neighbor']:

            sampler_kwargs, self.loader_kwargs = split_kwargs(
                self.kwargs,
                NeighborSampler,
            )
            sampler_kwargs.setdefault('share_memory',
                                      self.kwargs['num_workers'] > 0)
            self.graph_sampler: BaseSampler = NeighborSampler(
                data, **sampler_kwargs)

        elif graph_sampler is not None:
            sampler_kwargs, self.loader_kwargs = split_kwargs(
                self.kwargs,
                graph_sampler.__class__,
            )
            if len(sampler_kwargs) > 0:
                warnings.warn(
                    f"Ignoring the arguments "
                    f"{list(sampler_kwargs.keys())} in "
                    f"'{self.__class__.__name__}' since a custom "
                    f"'graph_sampler' was passed", stacklevel=2)
            self.graph_sampler = graph_sampler

        else:
            assert loader == 'full'
            self.loader_kwargs = self.kwargs


        self.eval_loader_kwargs = copy.copy(self.loader_kwargs)
        if eval_loader_kwargs is not None:
            if hasattr(self, 'graph_sampler'):
                self.eval_graph_sampler = copy.copy(self.graph_sampler)

                eval_sampler_kwargs, eval_loader_kwargs = split_kwargs(
                    eval_loader_kwargs,
                    self.graph_sampler.__class__,
                )
                for key, value in eval_sampler_kwargs.items():
                    setattr(self.eval_graph_sampler, key, value)

            self.eval_loader_kwargs.update(eval_loader_kwargs)

        elif hasattr(self, 'graph_sampler'):
            self.eval_graph_sampler = self.graph_sampler

        self.eval_loader_kwargs.pop('sampler', None)
        self.eval_loader_kwargs.pop('batch_sampler', None)

        if 'batch_sampler' in self.loader_kwargs:
            self.loader_kwargs.pop('batch_size', None)

    @property
    def train_shuffle(self) -> bool:
        pass

    def prepare_data(self) -> None:
        pass

    def full_dataloader(self, **kwargs: Any) -> torch.utils.data.DataLoader:
        pass

    def __repr__(self) -> str:
        kwargs = kwargs_repr(data=self.data, loader=self.loader, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


class LightningDataset(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        pred_dataset: Optional[Dataset] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            has_val=val_dataset is not None,
            has_test=test_dataset is not None,
            **kwargs,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.pred_dataset = pred_dataset

    def dataloader(self, dataset: Dataset, **kwargs: Any) -> DataLoader:
        pass

    def train_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass

    def predict_dataloader(self) -> DataLoader:
        pass

    def __repr__(self) -> str:
        kwargs = kwargs_repr(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            pred_dataset=self.pred_dataset,
            **self.kwargs,
        )
        return f'{self.__class__.__name__}({kwargs})'


class LightningNodeData(LightningData):
    def __init__(
        self,
        data: Union[Data, HeteroData],
        input_train_nodes: InputNodes = None,
        input_train_time: OptTensor = None,
        input_val_nodes: InputNodes = None,
        input_val_time: OptTensor = None,
        input_test_nodes: InputNodes = None,
        input_test_time: OptTensor = None,
        input_pred_nodes: InputNodes = None,
        input_pred_time: OptTensor = None,
        loader: str = 'neighbor',
        node_sampler: Optional[BaseSampler] = None,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if input_train_nodes is None:
            input_train_nodes = infer_input_nodes(data, split='train')

        if input_val_nodes is None:
            input_val_nodes = infer_input_nodes(data, split='val')
            if input_val_nodes is None:
                input_val_nodes = infer_input_nodes(data, split='valid')

        if input_test_nodes is None:
            input_test_nodes = infer_input_nodes(data, split='test')

        if input_pred_nodes is None:
            input_pred_nodes = infer_input_nodes(data, split='pred')

        super().__init__(
            data=data,
            has_val=input_val_nodes is not None,
            has_test=input_test_nodes is not None,
            loader=loader,
            graph_sampler=node_sampler,
            eval_loader_kwargs=eval_loader_kwargs,
            **kwargs,
        )

        self.input_train_nodes = input_train_nodes
        self.input_train_time = input_train_time
        self.input_train_id: OptTensor = None

        self.input_val_nodes = input_val_nodes
        self.input_val_time = input_val_time
        self.input_val_id: OptTensor = None

        self.input_test_nodes = input_test_nodes
        self.input_test_time = input_test_time
        self.input_test_id: OptTensor = None

        self.input_pred_nodes = input_pred_nodes
        self.input_pred_time = input_pred_time
        self.input_pred_id: OptTensor = None

    def dataloader(
        self,
        input_nodes: InputNodes,
        input_time: OptTensor = None,
        input_id: OptTensor = None,
        node_sampler: Optional[BaseSampler] = None,
        **kwargs: Any,
    ) -> torch.utils.data.DataLoader:
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        pass


class LightningLinkData(LightningData):
    def __init__(
        self,
        data: Union[Data, HeteroData],
        input_train_edges: InputEdges = None,
        input_train_labels: OptTensor = None,
        input_train_time: OptTensor = None,
        input_val_edges: InputEdges = None,
        input_val_labels: OptTensor = None,
        input_val_time: OptTensor = None,
        input_test_edges: InputEdges = None,
        input_test_labels: OptTensor = None,
        input_test_time: OptTensor = None,
        input_pred_edges: InputEdges = None,
        input_pred_labels: OptTensor = None,
        input_pred_time: OptTensor = None,
        loader: str = 'neighbor',
        link_sampler: Optional[BaseSampler] = None,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data=data,
            has_val=input_val_edges is not None,
            has_test=input_test_edges is not None,
            loader=loader,
            graph_sampler=link_sampler,
            eval_loader_kwargs=eval_loader_kwargs,
            **kwargs,
        )

        self.input_train_edges = input_train_edges
        self.input_train_labels = input_train_labels
        self.input_train_time = input_train_time
        self.input_train_id: OptTensor = None

        self.input_val_edges = input_val_edges
        self.input_val_labels = input_val_labels
        self.input_val_time = input_val_time
        self.input_val_id: OptTensor = None

        self.input_test_edges = input_test_edges
        self.input_test_labels = input_test_labels
        self.input_test_time = input_test_time
        self.input_test_id: OptTensor = None

        self.input_pred_edges = input_pred_edges
        self.input_pred_labels = input_pred_labels
        self.input_pred_time = input_pred_time
        self.input_pred_id: OptTensor = None

    def dataloader(
        self,
        input_edges: InputEdges,
        input_labels: OptTensor = None,
        input_time: OptTensor = None,
        input_id: OptTensor = None,
        link_sampler: Optional[BaseSampler] = None,
        **kwargs: Any,
    ) -> torch.utils.data.DataLoader:
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        pass




def infer_input_nodes(data: Union[Data, HeteroData], split: str) -> InputNodes:
    attr_name: Optional[str] = None
    if f'{split}_mask' in data:
        attr_name = f'{split}_mask'
    elif f'{split}_idx' in data:
        attr_name = f'{split}_idx'
    elif f'{split}_index' in data:
        attr_name = f'{split}_index'

    if attr_name is None:
        return None

    if isinstance(data, Data):
        return data[attr_name]
    if isinstance(data, HeteroData):
        input_nodes_dict = {
            node_type: store[attr_name]
            for node_type, store in data.node_items() if attr_name in store
        }
        if len(input_nodes_dict) != 1:
            raise ValueError(f"Could not automatically determine the input "
                             f"nodes of {data} since there exists multiple "
                             f"types with attribute '{attr_name}'")
        return list(input_nodes_dict.items())[0]
    return None


def kwargs_repr(**kwargs: Any) -> str:
    pass


def split_kwargs(
    kwargs: Dict[str, Any],
    sampler_cls: Type,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    r"""Splits keyword arguments into sampler and loader arguments."""
    sampler_args = inspect.signature(sampler_cls).parameters

    sampler_kwargs: Dict[str, Any] = {}
    loader_kwargs: Dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in sampler_args:
            sampler_kwargs[key] = value
        else:
            loader_kwargs[key] = value

    return sampler_kwargs, loader_kwargs
