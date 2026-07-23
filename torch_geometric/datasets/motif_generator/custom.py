from typing import Any, Optional

from torch_geometric.data import Data
from torch_geometric.datasets.motif_generator import MotifGenerator
from torch_geometric.utils import from_networkx


class CustomMotif(MotifGenerator):
    def __init__(self, structure: Any):
        super().__init__()

        self.structure: Optional[Data] = None

        if isinstance(structure, Data):
            self.structure = structure
        else:
            try:
                import networkx as nx
                if isinstance(structure, nx.Graph):
                    self.structure = from_networkx(structure)
            except ImportError:
                pass

        if self.structure is None:
            raise ValueError(f"Expected a motif structure of type "
                             f"'torch_geometric.data.Data' or 'networkx.Graph'"
                             f"(got {type(structure)})")

    def __call__(self) -> Data:
        assert isinstance(self.structure, Data)
        return self.structure
