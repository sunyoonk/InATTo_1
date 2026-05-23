"""LightGCN (He et al., SIGIR 2020).

Standard formulation:
    E^{(0)} = [user_embed; item_embed]                    learnable
    E^{(k+1)} = A_norm @ E^{(k)}                          linear, no nonlinearity
    E_final  = sum_{k=0..L} E^{(k)}                        layer aggregation
    score(u, i) = E_final[u] . E_final[n_users + i]

Loss: BPR
    L_bpr = -mean log sigma( score(u, i_pos) - score(u, i_neg) )
    L2 reg on the embedding parameters (controlled by `reg_weight`).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_size: int = 256,
        layer_num: int = 3,
    ):
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_size = int(embedding_size)
        self.layer_num = int(layer_num)

        self.user_embeds = nn.Parameter(torch.empty(n_users, embedding_size))
        self.item_embeds = nn.Parameter(torch.empty(n_items, embedding_size))
        nn.init.xavier_uniform_(self.user_embeds)
        nn.init.xavier_uniform_(self.item_embeds)

    def propagate(self, adj_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (user_final, item_final), shapes [n_users, d] and [n_items, d]."""
        e = torch.cat([self.user_embeds, self.item_embeds], dim=0)
        layers = [e]
        for _ in range(self.layer_num):
            e = torch.sparse.mm(adj_norm, layers[-1])
            layers.append(e)
        # LightGCN final: simple sum (or mean) of layer outputs
        out = torch.stack(layers, dim=0).sum(dim=0)
        return out[: self.n_users], out[self.n_users :]

    @staticmethod
    def bpr_loss(
        u_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        neg_embed: torch.Tensor,
    ) -> torch.Tensor:
        pos_score = (u_embed * pos_embed).sum(dim=-1)
        neg_score = (u_embed * neg_embed).sum(dim=-1)
        return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
