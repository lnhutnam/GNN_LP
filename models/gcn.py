import torch
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import MessagePassing


class GraphConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConv, self).__init__(aggr=None)
        self.mlp1 = MLP([in_channels + out_channels, hidden_channels, out_channels])
        self.mlp2 = MLP([in_channels, hidden_channels, out_channels])

    def forward(self, x, edge_index, edge_weight, size):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

    def message(self, x, edge_weight):
        reshaped = edge_weight[:, :, None].reshape(
            (edge_weight.shape[0] * edge_weight.shape[1]) // self.mlp2(x[1]).shape[0],
            self.mlp2(x[1]).shape[0],
            1,
        )
        a = torch.mul(reshaped, self.mlp2(x[1]))
        a = a.reshape(
            x[0].shape[0], (a.shape[0] * a.shape[1]) // x[0].shape[0], a.shape[2]
        )
        return a

    def aggregate(self, inputs):
        return torch.sum(inputs, dim=1)

    def update(self, aggr_out, x):
        return self.mlp1(torch.cat((x[0], aggr_out), dim=1))
