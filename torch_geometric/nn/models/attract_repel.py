import torch
import torch.nn.functional as F


class ARLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=None,
                 num_layers=2, dropout=0.0, attract_ratio=0.5):
        super().__init__()

        if out_channels is None:
            out_channels = hidden_channels

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        if not 0 <= attract_ratio <= 1:
            raise ValueError(
                f"attract_ratio must be between 0 and 1, got {attract_ratio}")

        self.attract_ratio = attract_ratio
        self.attract_dim = int(out_channels * attract_ratio)
        self.repel_dim = out_channels - self.attract_dim

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

        self.lin_attract = torch.nn.Linear(hidden_channels, self.attract_dim)
        self.lin_repel = torch.nn.Linear(hidden_channels, self.repel_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_attract.reset_parameters()
        self.lin_repel.reset_parameters()

    def encode(self, x, *args, **kwargs):
        """Encode node features into attract-repel embeddings.

        Args:
            x (torch.Tensor): Node feature matrix of shape
                :obj:`[num_nodes, in_channels]`.
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        """
        for lin in self.lins:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        attract_x = self.lin_attract(x)
        repel_x = self.lin_repel(x)

        return attract_x, repel_x

    def decode(self, attract_z, repel_z, edge_index):
        """Decode edge scores from attract-repel embeddings.

        Args:
            attract_z (torch.Tensor): Attract embeddings of shape
                :obj:`[num_nodes, attract_dim]`.
            repel_z (torch.Tensor): Repel embeddings of shape
                :obj:`[num_nodes, repel_dim]`.
            edge_index (torch.Tensor): Edge indices of shape
                :obj:`[2, num_edges]`.

        Returns:
            torch.Tensor: Edge prediction scores.
        """
        row, col = edge_index
        attract_z_row = attract_z[row]
        attract_z_col = attract_z[col]
        repel_z_row = repel_z[row]
        repel_z_col = repel_z[col]

        attract_score = torch.sum(attract_z_row * attract_z_col, dim=1)
        repel_score = torch.sum(repel_z_row * repel_z_col, dim=1)

        return attract_score - repel_score

    def forward(self, x, edge_index):
        """Forward pass for link prediction.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge indices to predict.

        Returns:
            torch.Tensor: Predicted edge scores.
        """
        attract_z, repel_z = self.encode(x)

        return torch.sigmoid(self.decode(attract_z, repel_z, edge_index))

    def calculate_r_fraction(self, attract_z, repel_z):
        pass
