import torch
from torch_geometric.data import Data


class BipartiteData(Data):
    def __init__(self, x_s, x_t, edge_index, edge_weight, num_nodes):
        super().__init__(num_nodes=num_nodes)
        self.x_s = x_s
        self.x_t = x_t
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
