from typing import Optional

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


@functional_transform('laplacian_lambda_max')
class LaplacianLambdaMax(BaseTransform):
    def __init__(
        self,
        normalization: Optional[str] = None,
        is_undirected: bool = False,
    ):
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        self.normalization = normalization
        self.is_undirected = is_undirected

    def forward(self, data: Data) -> Data:
        from scipy.sparse.linalg import eigs, eigsh

        assert data.edge_index is not None
        num_nodes = data.num_nodes

        edge_weight = data.edge_attr
        if edge_weight is not None and edge_weight.numel() != data.num_edges:
            edge_weight = None

        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            edge_weight,
            self.normalization,
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        eig_fn = eigs
        if self.is_undirected and self.normalization != 'rw':
            eig_fn = eigsh

        lambda_max = eig_fn(L, k=1, which='LM', return_eigenvectors=False)
        data.lambda_max = lambda_max.real.item()

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(normalization={self.normalization})'
