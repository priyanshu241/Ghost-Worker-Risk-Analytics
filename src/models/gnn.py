import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class WorkerRiskGCN(torch.nn.Module):
    """
    Three-layer Graph Convolutional Network for predicting displacement risk.

    Takes node features and graph connectivity as input and outputs a risk
    score between 0 and 1 per node via sigmoid activation.

    Args:
        in_channels      number of input features per node
        hidden_channels  size of the two hidden layers (default 64)
        out_channels     1 for a single risk score output
        dropout          dropout rate applied during training (default 0.3)
    """

    def __init__(self, in_channels, hidden_channels=64, out_channels=1, dropout=0.3):
        super(WorkerRiskGCN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1   = BatchNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2   = BatchNorm(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        return torch.sigmoid(x)

    def get_embeddings(self, x, edge_index):
        """
        Returns the hidden representations from the second layer.
        Useful for visualisation and downstream clustering tasks.
        """
        self.eval()
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(x, edge_index)))
            x = F.relu(self.bn2(self.conv2(x, edge_index)))
        return x
