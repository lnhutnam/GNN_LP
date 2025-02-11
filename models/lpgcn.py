import torch
import torch.nn as nn

from torch_geometric.nn import MLP
from torch_geometric.loader import DataLoader

import numpy as np

from models.gcn import GraphConv
from utils.data import BipartiteData


class LPGCN(nn.Module):
    def __init__(self, num_constraints, num_variables, num_layers=5):
        super().__init__()

        self.num_constraints = num_constraints
        self.num_variables = num_variables
        self.num_layers = num_layers

        ints = np.random.randint(1, 10, size=self.num_layers)
        dims = [2**i for i in ints]

        self.fv_in = MLP([2, 32, dims[0]])
        self.fw_in = MLP([3, 32, dims[0]])

        self.cv = nn.ModuleList(
            [GraphConv(dims[l - 1], 32, dims[l]) for l in range(1, self.num_layers)]
        )
        self.cw = nn.ModuleList(
            [GraphConv(dims[l - 1], 32, dims[l]) for l in range(1, self.num_layers)]
        )

        self.f_out = MLP([2 * dims[self.num_layers - 1], 32, 1])

        self.fw_out = MLP([3 * dims[self.num_layers - 1], 32, 1])

    def construct_graph(self, c, A, b, constraints, l, u):
        hv = torch.cat((b.unsqueeze(2), constraints.unsqueeze(2)), dim=2)

        hw = torch.cat((c.unsqueeze(2), l.unsqueeze(2), u.unsqueeze(2)), dim=2)

        E = A

        return hv, hw, E

    def init_features(self, hv, hw):
        hv_0 = []
        for i in range(self.num_constraints):
            hv_0.append(self.fv_in(hv[:, i]))

        hw_0 = []
        for j in range(self.num_variables):
            hw_0.append(self.fw_in(hw[:, j]))

        hv = torch.stack(hv_0, dim=1)
        hw = torch.stack(hw_0, dim=1)

        return hv, hw

    def convs(self, hv, hw, edge_index, E, layer, batch_size):
        hv_l = self.cv[layer](
            (hv, hw),
            edge_index,
            E,
            (self.num_constraints * batch_size, self.num_variables),
        )

        hw_l = self.cw[layer](
            (hw, hv),
            torch.flip(edge_index, dims=[1, 0]),
            E.T,
            (self.num_variables, self.num_constraints * batch_size),
        )

        return hv_l, hw_l

    def single_output(self, hv, hw):
        y_out = self.f_out(torch.cat((torch.sum(hv, 1), torch.sum(hw, 1)), dim=1))
        return y_out

    def sol_output(self, hv, hw):
        sol = []
        for j in range(self.num_variables):
            joint = torch.cat((torch.sum(hv, 1), torch.sum(hw, 1), hw[:, j]), dim=1)
            sol.append(self.fw_out(joint))

        sol = torch.stack(sol, dim=1)
        return sol[:, :, 0]

    def forward(self, c, A, b, constraints, l, u, edge_index, phi):

        hv, hw, E = self.construct_graph(c, A, b, constraints, l, u)
        hv, hw = self.init_features(hv, hw)

        batch_size = hv.shape[0]

        graphs = [
            BipartiteData(
                x_s=hv[i],
                x_t=hw[i],
                edge_index=edge_index[i].T,
                edge_weight=E[i],
                num_nodes=self.num_variables + self.num_constraints,
            )
            for i in range(hv.shape[0])
        ]
        loader = DataLoader(graphs, batch_size=batch_size)
        batch = next(iter(loader))

        hv = batch.x_s
        hw = batch.x_t
        edge_index = batch.edge_index
        E = batch.edge_weight

        for l in range(self.num_layers - 1):
            hv, hw = self.convs(hv, hw, edge_index, E, l, batch_size)

        hv = hv.reshape(batch_size, hv.shape[0] // batch_size, hv.shape[1])
        hw = hw.reshape(batch_size, hw.shape[0] // batch_size, hw.shape[1])

        if phi == "feas":
            output = self.single_output(hv, hw)
            bins = [1 if elem >= 1 / 2 else 0 for elem in output]
            return torch.tensor(bins, dtype=torch.float32, requires_grad=True)

        elif phi == "obj":
            return self.single_output(hv, hw)

        elif phi == "sol":
            return self.sol_output(hv, hw)

        else:
            return "Please, choose one type of function: feas, obj or sol"
