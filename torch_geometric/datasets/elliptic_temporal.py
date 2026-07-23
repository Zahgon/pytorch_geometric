from typing import Any, Callable, Optional, Tuple

from torch_geometric.datasets import EllipticBitcoinDataset


class EllipticBitcoinTemporalDataset(EllipticBitcoinDataset):
    def __init__(
        self,
        root: str,
        t: int,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        if t < 1 or t > 49:
            raise ValueError("'t' needs to be between 1 and 49")

        self.t = t
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)

    @property
    def processed_file_names(self) -> str:
        pass

    def _process_df(self, feat_df: Any, edge_df: Any,
                    class_df: Any) -> Tuple[Any, Any, Any]:

        feat_df = feat_df[feat_df['time_step'] == self.t]

        mask = edge_df['txId1'].isin(feat_df['txId'].values)
        edge_df = edge_df[mask]

        class_df = class_df.merge(feat_df[['txId']], how='right',
                                  left_on='txId', right_on='txId')

        return feat_df, edge_df, class_df
