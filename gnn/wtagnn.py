import math
import torch as th
import torch.nn as nn


def wtagnn_msg(edge):
    nb_ef = edge.data["ef"]
    return {"nb_ef": nb_ef}


def wtagnn_reduce(node):
    nb_ef = th.mean(node.mailbox["nb_ef"], 1)
    return {"nb_ef": nb_ef}


class NodeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1.0 / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        nf = nodes.data["nf"]
        nb_ef = nodes.data["nb_ef"]

        if self.bias is not None:
            nf = nf + self.bias
        if self.activation:
            nf = self.activation(nf)
        return {"nf": nf, "nb_ef": nb_ef}


class EdgeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(EdgeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()
        self.my_dense = nn.Linear(out_feats * 2, out_feats)

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1.0 / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, edges):
        ### msg pass strategy: ef = ef + nb_ef + srcdst_nf
        tmp_ef = edges.data["ef"] + edges.dst["nb_ef"]  # mean/max-fy at msg_reduce
        srcdst_nf = (edges.src["nf"] + edges.dst["nf"]) / 2
        ef = self.my_dense(th.cat([tmp_ef, srcdst_nf], 1))

        if self.bias is not None:
            ef = ef + self.bias
        if self.activation:
            ef = self.activation(ef)

        return {"ef": ef}


class WTAGNNLayer(nn.Module):
    def __init__(
        self, g, in_feats_node, in_feats_edge, out_feats, activation, dropout, bias=True
    ):
        super(WTAGNNLayer, self).__init__()
        self.g = g
        # print("in_feats_node  out_feats", in_feats_node, out_feats)
        self.weight_node = nn.Parameter(th.Tensor(in_feats_node, out_feats))
        self.weight_edge = nn.Parameter(th.Tensor(in_feats_edge, out_feats))
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.0
        self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.edge_update = EdgeApplyModule(out_feats, activation, bias)
        self.reset_parameters()
        self.PRINT = False

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight_node.size(1))
        self.weight_node.data.uniform_(-stdv, stdv)

        stdv_edge = 1.0 / math.sqrt(self.weight_edge.size(1))
        self.weight_edge.data.uniform_(-stdv_edge, stdv_edge)

    def forward(self, edge_subgraph, nf, ef):
        if self.dropout:
            nf = self.dropout(nf)
        # print(nf.shape, self.weight_node.shape)
        edge_subgraph.ndata["nf"] = th.mm(nf, self.weight_node)
        edge_subgraph.edata["ef"] = th.mm(ef, self.weight_edge)

        if self.PRINT:
            print("After mm, nf: ", edge_subgraph.ndata["nf"])
        if self.PRINT:
            print("After mm, ef: ", edge_subgraph.edata["ef"])

        edge_subgraph.update_all(wtagnn_msg, wtagnn_reduce)

        if self.PRINT:
            print("After update_all, nf: ", edge_subgraph.ndata["nf"])
        if self.PRINT:
            print("After update_all, nb_ef: ", edge_subgraph.ndata["nb_ef"])
        if self.PRINT:
            print("After update_all, ef: ", edge_subgraph.edata["ef"])
        edge_subgraph.apply_nodes(func=self.node_update)

        edge_subgraph.apply_edges(func=self.edge_update)
        if self.PRINT:
            print("After apply_nodes, nf: ", edge_subgraph.ndata["nf"])
        if self.PRINT:
            print("After apply_nodes, nb_ef: ", edge_subgraph.ndata["nb_ef"])
        if self.PRINT:
            print("After apply_edges, ef: ", edge_subgraph.edata["ef"])

        nf = edge_subgraph.ndata.pop("nf")
        ef = edge_subgraph.edata.pop("ef")
        edge_subgraph.ndata.pop("nb_ef")
        return nf, ef


class WTAGNN(nn.Module):
    def __init__(
        self,
        g,
        input_node_feat_size,
        input_edge_feat_size,
        n_hidden,
        n_classes,
        n_layers,
        activation,
        dropout,
    ):
        super(WTAGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            WTAGNNLayer(
                g,
                input_node_feat_size,
                input_edge_feat_size,
                n_hidden,
                activation,
                dropout,
            )
        )
        for i in range(n_layers - 1):
            self.layers.append(
                WTAGNNLayer(g, n_hidden, n_hidden, n_hidden, activation, dropout)
            )
        self.layers.append(WTAGNNLayer(g, n_hidden, n_hidden, n_classes, None, dropout))

    def forward(self, edge_subgraph, nf, ef):
        for layer in self.layers:
            nf, ef = layer(edge_subgraph, nf, ef)
        return nf, ef
