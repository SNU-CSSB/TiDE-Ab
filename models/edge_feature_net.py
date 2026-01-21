import torch
from torch import nn

class EdgeFeatureNet(nn.Module):
    
    def __init__(self, module_cfg):
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg
    
        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim
        self.relpos_k = self._cfg.relpos_k
        self.relpos_n_bin = 2 * self._cfg.relpos_k + 2
    
        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        
        if self._cfg.ori_rel_pos:
            self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)
        else:
            self.linear_relpos = nn.Linear(self.relpos_n_bin, self.feat_dim, bias=False)
    
        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
        if self._cfg.use_2d_hotspot: total_edge_feats += 1
        if self._cfg.same_diff_disto: 
            total_edge_feats -= self._cfg.num_bins
            total_edge_feats += 1
        
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )
    
    def embed_relpos(self, r, m, is_same_chain=None):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        # Input: [b, n_res]
        # [b, n_res, n_res]
        d = r[:, :, None] - r[:, None, :]
        if self._cfg.ori_rel_pos:
            pos_emb = get_index_embedding(d, self._cfg.feat_dim, max_len=2056)
            return self.linear_relpos(pos_emb)
        else:
            d_same_chain = torch.clip(r[:, :, None] - r[:, None, :] + self.relpos_k, 0, 2 * self.relpos_k)
            d_diff_chain = torch.ones_like(d_same_chain) * (2 * self.relpos_k + 1) 
            d = d_same_chain * is_same_chain + d_diff_chain * ~is_same_chain
            oh = nn.functional.one_hot(d.long(), num_classes=self.relpos_n_bin).float()
            return self.linear_relpos(oh)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])
    
    def forward(self, s, dist_feats, sc_feats, p_mask, res_idx, res_mask, hotspot_2d=None, is_same_chain=None, disto_cond_mask=None):
        
        # Input: [b, n_res, c_s]
        num_batch, num_res, _ = s.shape
    
        # [b, n_res, c_p]
        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)
        
        relpos_feats = self.embed_relpos(res_idx, res_mask, is_same_chain)

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats]
        if not self._cfg.same_diff_disto:
            all_edge_feats.append(sc_feats)
        if self._cfg.use_2d_hotspot:
            all_edge_feats.append(hotspot_2d)
        if self._cfg.same_diff_disto:
            all_edge_feats.append(disto_cond_mask)
            
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats