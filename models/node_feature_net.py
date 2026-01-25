import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb

        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb + 3 + 21 + 1
        self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(timesteps[:, 0], self.c_timestep_emb, max_positions=2056)[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, ts_, node_mask, res_idx, chain_idx=None, aatype_idx=None, hotspot_1d=None):
        b, num_res, device = node_mask.shape[0], node_mask.shape[1], node_mask.device

        # [b, n_res, c_pos_emb]
        res_idx_emb = get_index_embedding(res_idx, self.c_pos_emb, max_len=2056)
        input_feats = [res_idx_emb, self.embed_t(ts_, node_mask), chain_idx, aatype_idx, hotspot_1d]

        return self.linear(torch.cat(input_feats, dim=-1))
