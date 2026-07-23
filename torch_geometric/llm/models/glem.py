from typing import List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn.models import GraphSAGE, basic_gnn


def deal_nan(x):
    if isinstance(x, torch.Tensor):
        x = x.clone()
        x[torch.isnan(x)] = 0.0
    return x


class GLEM(torch.nn.Module):
    def __init__(
        self,
        lm_to_use: str = 'prajjwal1/bert-tiny',
        gnn_to_use: basic_gnn = GraphSAGE,
        out_channels: int = 47,
        gnn_loss: Optional[nn.Module] = None,
        lm_loss: Optional[nn.Module] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        lm_dtype: torch.dtype = torch.bfloat16,
        lm_use_lora: bool = True,
        lora_target_modules: Optional[Union[List[str], str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        if gnn_loss is None:
            gnn_loss = nn.CrossEntropyLoss(reduction='mean')
        if lm_loss is None:
            lm_loss = nn.CrossEntropyLoss(reduction='mean')
        if device is None:
            device = torch.device('cpu')

        self.device = device
        self.lm_loss = lm_loss
        self.gnn = gnn_to_use
        self.gnn_loss = gnn_loss
        self.alpha = alpha
        self.beta = beta
        self.gnn_loss = gnn_loss
        self.lm = lm_to_use

        if lm_to_use == "prajjwal1/bert-tiny":
            from transformers import BertForSequenceClassification
            model_class = BertForSequenceClassification
        else:
            from transformers import AutoModelForSequenceClassification
            model_class = AutoModelForSequenceClassification

        self.lm = model_class.from_pretrained(
            lm_to_use,
            num_labels=out_channels,
            torch_dtype=lm_dtype,
            offload_folder="offload",
            trust_remote_code=True,
        )

        if lm_use_lora:
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            print("Training LM with LORA!")
            self.lm = prepare_model_for_kbit_training(self.lm)
            config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16,
                                lora_alpha=16, lora_dropout=0.05, bias="none",
                                target_modules=lora_target_modules)
            self.lm = get_peft_model(self.lm, config)
            self.lm.print_trainable_parameters()
        self.lm.config.pad_token_id = self.lm.config.eos_token_id
        self.lm_device = self.lm.device

        if self.lm.num_labels != self.gnn.out_channels:
            raise ValueError('''The output channel of language model \
                             and gnn should be the same''')

    def pre_train_gnn(self, train_loader: NeighborLoader,
                      optimizer: torch.optim.Optimizer, num_epochs: int,
                      patience: int, ext_pseudo_labels: torch.Tensor = None,
                      is_augmented: bool = False, verbose: bool = True):
        pass

    def pre_train_lm(self, train_loader: DataLoader,
                     optimizer: torch.optim.Optimizer, num_epochs: int,
                     patience: int, ext_pseudo_labels: torch.Tensor = None,
                     is_augmented: bool = False, verbose: bool = True):
        pass

    def train(self, em_phase: str, train_loader: Union[DataLoader,
                                                       NeighborLoader],
              optimizer: torch.optim.Optimizer, pseudo_labels: torch.Tensor,
              epoch: int, is_augmented: bool = False, verbose: bool = False):
        r"""GLEM training step, EM steps.

        Args:
            em_phase(str): 'gnn' or 'lm' choose which phase you are training on
            train_loader(Union[DataLoader, NeighborLoader]): use DataLoader for
                lm training, include tokenized data, labels is_gold mask.
                use NeighborLoader for gnn training, include x, edge_index.
            optimizer (torch.optim.Optimizer): optimizer for training
            pseudo_labels(torch.Tensor): the predicted labels used as pseudo
                labels
            epoch (int): current epoch
            is_augmented (bool): will use pseudo_labels or not
            verbose (bool): print training progress bar or not

        Returns:
            acc (float): training accuracy
            loss (float): loss value
        """
        if pseudo_labels is not None:
            pseudo_labels = pseudo_labels.to(self.device)
        if em_phase == 'gnn':
            acc, loss = self.train_gnn(train_loader, optimizer, epoch,
                                       pseudo_labels, is_augmented, verbose)
        if em_phase == 'lm':
            acc, loss = self.train_lm(train_loader, optimizer, epoch,
                                      pseudo_labels, is_augmented, verbose)
        return acc, loss

    def train_lm(self, train_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, epoch: int,
                 pseudo_labels: torch.Tensor = None,
                 is_augmented: bool = False, verbose: bool = True):
        r"""Language model Training in every epoch.

        Args:
            train_loader (loader.dataloader.DataLoader): text token dataloader
            optimizer (torch.optim.Optimizer): model optimizer
            epoch (int): current train epoch
            pseudo_labels (torch.Tensor): 1-D tensor, predictions from gnn
            is_augmented (bool): train with pseudo labels or not
            verbose (bool): print training progress bar or not

        Returns:
            approx_acc (torch.tensor): training accuracy
            loss (torch.float): loss value

        """
        all_out = []
        total_loss = total_correct = 0
        num_nodes = train_loader.dataset.indices.size(0)
        self.lm.train()
        if verbose:
            pbar = tqdm(total=num_nodes)
            pbar.set_description(f'Epoch {epoch:02d}')
        for batch in train_loader:
            inputs = {k: v.to(self.device) for k, v in batch['input'].items()}
            out = self.lm(**inputs).logits
            labels = batch['labels'].to(self.device).squeeze()
            if is_augmented:
                pl_batch = pseudo_labels[batch['n_id']].to(self.device)
            else:
                pl_batch = None
            loss = self.loss(out, labels, self.lm_loss,
                             batch['is_gold'].to(self.device), pl_batch,
                             self.alpha, is_augmented)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_out.append(out)
            total_correct += int(out.argmax(dim=-1).eq(labels).sum())
            total_loss += float(loss.detach())
            if verbose:
                pbar.update(batch['n_id'].size(0))

        all_out = torch.cat(all_out, dim=0)
        approx_acc = total_correct / num_nodes
        loss = total_loss / len(train_loader)
        if verbose:
            pbar.close()
        print(f'Epoch {epoch:02d} Loss: {loss:.4f} '
              f'Approx. Train: {approx_acc:.4f}')
        return approx_acc, loss

    def train_gnn(self, train_loader: NeighborLoader,
                  optimizer: torch.optim.Optimizer, epoch: int,
                  pseudo_labels: torch.Tensor = None,
                  is_augmented: bool = False, verbose: bool = True):
        r"""GNN training step in every epoch.

        Args:
            train_loader (loader.NeighborLoader): gnn Neighbor node loader
            optimizer (torch.optim.Optimizer): model optimizer
            epoch (int): current train epoch
            pseudo_labels(torch.tensor): 1-D tensor, predictions from lm
            is_augmented(bool): use pseudo labeled node or not
            verbose (bool): print training progress or not

        Returns:
            approx_acc (torch.tensor): training accuracy
            loss (torch.float): loss value
        """
        self.gnn.train()
        num_nodes = train_loader.input_nodes.size(0)
        if verbose:
            pbar = tqdm(total=num_nodes)
            pbar.set_description(f'Epoch {epoch:02d}')
        total_loss = total_correct = 0
        all_out = []
        for batch in train_loader:
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index)[:batch.batch_size]
            all_out.append(out)
            labels = batch.y[:batch.batch_size].squeeze()
            is_gold_batch = batch.is_gold[:batch.batch_size].squeeze()
            if is_augmented and pseudo_labels is not None:
                pl_batch = pseudo_labels[batch.n_id[:batch.batch_size]]
            else:
                pl_batch = None
            loss = self.loss(out, labels, self.gnn_loss, is_gold_batch,
                             pl_batch, self.beta, is_augmented)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach())
            total_correct += int(out.argmax(dim=-1).eq(labels).sum())
            if verbose:
                pbar.update(batch.batch_size)

        all_out = torch.cat(all_out, dim=0)
        loss = total_loss / len(train_loader)
        approx_acc = total_correct / num_nodes
        if verbose:
            pbar.close()
        print(f'Epoch: {epoch:02d} Loss: {loss:.4f} '
              f'Approx. Train: {approx_acc:.4f}')
        return approx_acc, loss

    @torch.no_grad()
    def inference(self, em_phase: str, data_loader: Union[NeighborLoader,
                                                          DataLoader],
                  verbose: bool = False):
        r"""GLEM inference step.

        Args:
            em_phase(str): 'gnn' or 'lm'
            data_loader(dataloader or Neighborloader):
                dataloader: for lm training, include tokenized data
                nodeloader: for gnn training, include x, edge_index
            verbose(bool): print inference progress or not

        Returns:
            out (torch.Tensor): n * m tensor, m is number of classes,
                n is number of nodes
        """
        out = None
        if em_phase == 'gnn':
            self.gnn.eval()
            out = self.inference_gnn(data_loader, verbose)
        elif em_phase == 'lm':
            self.lm.eval()
            out = self.inference_lm(data_loader, verbose)
        return out

    @torch.no_grad()
    def inference_lm(self, data_loader: DataLoader, verbose: bool = True):
        r"""LM inference step.

        Args:
            data_loader (Dataloader): include token, labels, and gold mask
            verbose (bool): print progress bar or not

        Returns:
            preds (tensor): prediction from GNN, convert to pseudo labels
                by preds.argmax(dim=-1).unsqueeze(1)
        """
        if verbose:
            pbar = tqdm(total=data_loader.dataset._data.num_nodes)
            pbar.set_description('LM inference stage')
        self.lm.eval()
        preds = []
        for batch in data_loader:
            inputs = {k: v.to(self.device) for k, v in batch['input'].items()}
            logits = self.lm(**inputs).logits
            preds.append(logits)
            if verbose:
                pbar.update(batch['n_id'].size(0))
        if verbose:
            pbar.close()
        preds = torch.cat(preds)
        return preds

    @torch.no_grad()
    def inference_gnn(self, data_loader: NeighborLoader, verbose: bool = True):
        r"""GNN inference step.

        Args:
            data_loader(NeighborLoader): include x, edge_index,
            verbose (bool): print progress bar or not

        Returns:
            preds (tensor): prediction from GNN,
                convert to pseudo labels by preds.argmax(dim=-1).unsqueeze(1)
        """
        if verbose:
            pbar = tqdm(total=data_loader.data.num_nodes)
            pbar.set_description('GNN inference stage')
        preds = []
        self.gnn.eval()
        for batch in data_loader:
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index)[:batch.batch_size]
            preds.append(out)
            if verbose:
                pbar.update(batch.batch_size)
        if verbose:
            pbar.close()
        preds = torch.cat(preds, dim=0)
        return preds

    def loss(self, logits: torch.Tensor, labels: torch.Tensor,
             loss_func: torch.nn.functional, is_gold: torch.Tensor,
             pseudo_labels: torch.Tensor = None, pl_weight: float = 0.5,
             is_augmented: bool = True):
        r"""Core function of variational EM inference, this function is aming
        on combining loss value on gold(original train) and loss value on
        pseudo labels.

        Reference:
        <https://github.com/AndyJZhao/GLEM/blob/main/src/models/GLEM/GLEM_utils.py> # noqa

        Args:
            logits(torch.tensor): predict results from LM or GNN
            labels(torch.tensor): combined node labels from ground truth and
                pseudo labels(if provided)
            loss_func(torch.nn.modules.loss): loss function for classification
            is_gold(tensor): a tensor with bool value that mask ground truth
                    label and during training, thus ~is_gold mask pseudo labels
            pseudo_labels(torch.tensor): predictions from other model
            pl_weight: the pseudo labels used in E-step and M-step optimization
                        alpha in E-step, beta in M-step respectively
            is_augmented: use EM or just train GNN and LM with gold data

        """
        if is_augmented and (sum(~is_gold) > 0):
            mle_loss = deal_nan(loss_func(logits[is_gold], labels[is_gold]))
            pseudo_label_loss = deal_nan(
                loss_func(logits[~is_gold], pseudo_labels[~is_gold]))
            loss = pl_weight * pseudo_label_loss + (1 - pl_weight) * mle_loss
        else:
            loss = loss_func(logits, labels)
        return loss
