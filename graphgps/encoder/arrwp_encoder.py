import torch
from torch import nn

from torch_geometric.utils import coalesce


class ARRWPLinearNodeEncoder(nn.Module):
    """
    FC_1(ARRWP) + FC_2 (node_attr)
    note: FC_2 is supposed to be applied
    """

    def __init__(
        self, in_dim, emb_dim,
        norm_type=None, use_bias=False,
        mx_name="arrwp",
    ):
        super().__init__()

        self.mx_name = mx_name

        self.fc = nn.Linear(in_dim, emb_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(emb_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(emb_dim)
        else:
            self.norm = None

    def forward(self, batch):
        arrwp = batch[f"node_{self.mx_name}"]
        arrwp = self.fc(arrwp)

        if "x" in batch:
            batch.x = batch.x + arrwp
        else:
            batch.x = arrwp

        if self.norm:
            batch.x = self.norm(batch.x)

        return batch
    

class ARRWPLinearEdgeEncoder(torch.nn.Module):
    """
    FC_1(ARRWP) + FC_2(edge_attr)
    notes:
        - Batchnorm/Layernorm might ruin some properties of
        encoding on providing shortest-path distance info
        - FC_2 is supposed to be applied
    """

    def __init__(
        self, in_dim, emb_dim,
        norm_type=None, use_bias=False,
        mx_name="arrwp",
    ):
        super().__init__()

        self.mx_name = mx_name

        self.fc = nn.Linear(in_dim, emb_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(emb_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(emb_dim)
        else:
            self.norm = None

    def forward(self, batch):
        if batch.edge_attr is None:
            batch.edge_attr = batch.edge_index.new_zeros(
                batch.edge_index.size(1), batch.arrwp_attr.size(1),
            )

        # for troubles with PyG 2.0.4 fixing
        batch.edge_index = torch.cat(
            (batch.edge_index, batch.edge_attr.T)
        )

        data_list = batch.to_data_list()
        for data in data_list:
            arrwp_index = data[f"edge_{self.mx_name}_index"]
            arrwp_attr = data[f"edge_{self.mx_name}_attr"]
            arrwp_attr = self.fc(arrwp_attr)

            edge_index = data.edge_index[:2].long()
            edge_attr = data.edge_index[2:].T

            upd_edge_index, upd_edge_attr = coalesce(
                torch.cat([edge_index, arrwp_index], dim=1),
                torch.cat([edge_attr, arrwp_attr], dim=0),
                reduce="add",
            )

            if self.norm:
                upd_edge_attr = self.norm(upd_edge_attr)

            data.edge_index, data.edge_attr =\
                upd_edge_index, upd_edge_attr

        upd_batch = batch.from_data_list(data_list)
        for key in batch.keys:
            if key not in upd_batch:
                upd_batch[key] = batch[key]

        return upd_batch