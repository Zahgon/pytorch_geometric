import os.path as osp
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from torch_geometric.io import fs

try:
    from huggingface_hub import ModelHubMixin, hf_hub_download
except ImportError:
    ModelHubMixin = object
    hf_hub_download = None

CONFIG_NAME = 'config.json'
MODEL_HUB_ORGANIZATION = "pytorch_geometric"
MODEL_WEIGHTS_NAME = 'model.pth'
TAGS = ['graph-machine-learning']


class PyGModelHubMixin(ModelHubMixin):
    def __init__(self, model_name: str, dataset_name: str, model_kwargs: Dict):
        ModelHubMixin.__init__(self)

        self.model_config = {
            k: v
            for k, v in model_kwargs.items() if type(v) in [str, int, float]
        }
        self.model_name = model_name
        self.dataset_name = dataset_name

    def construct_model_card(self, model_name: str, dataset_name: str) -> Any:
        pass

    def _save_pretrained(self, save_directory: Union[Path, str]):
        pass

    def save_pretrained(self, save_directory: Union[str, Path],
                        push_to_hub: bool = False,
                        repo_id: Optional[str] = None, **kwargs):
        pass

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        local_files_only,
        token,
        proxies=None,
        resume_download=False,
        dataset_name='',
        model_name='',
        map_location='cpu',
        strict=False,
        **model_kwargs,
    ):
        pass

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **model_kwargs,
    ) -> Any:
        r"""Downloads and instantiates a model from the HuggingFace hub.

        Args:
            pretrained_model_name_or_path (str): Can be either:

                - The :obj:`model_id` of a pretrained model hosted inside the
                  HuggingFace hub.

                - You can add a :obj:`revision` by appending :obj:`@` at the
                  end of :obj:`model_id` to load a specific model version.

                - A path to a directory containing the saved model weights.

                - :obj:`None` if you are both providing the configuration
                  :obj:`config` and state dictionary :obj:`state_dict`.

            force_download (bool, optional): Whether to force the
                (re-)download of the model weights and configuration files,
                overriding the cached versions if they exist.
                (default: :obj:`False`)
            token (str or bool, optional): The token to use as HTTP bearer
                authorization for remote files. If set to :obj:`True`, will use
                the token generated when running :obj:`transformers-cli login`
                (stored in :obj:`~/.huggingface`). It is **required** if you
                want to use a private model. (default: :obj:`None`)
            cache_dir (str, optional): The path to a directory in which a
                downloaded model configuration should be cached if the
                standard cache should not be used. (default: :obj:`None`)
            local_files_only (bool, optional): Whether to only look at local
                files, *i.e.* do not try to download the model.
                (default: :obj:`False`)
            **model_kwargs: Additional keyword arguments passed to the
                model during initialization.
        """
        return super().from_pretrained(
            pretrained_model_name_or_path,
            force_download=force_download,
            use_auth_token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs,
        )
