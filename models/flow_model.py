import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models import ipa_pytorch
from models.utils import calc_distogram
from data import utils as du


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)
        if self._model_conf.pred_aatype:
            self.aa_pred_head = nn.Linear(self._model_conf.node_embed_size, 21)

        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(d_model=tfmr_in, nhead=self._ipa_conf.seq_tfmr_num_heads,
                                                          dim_feedforward=tfmr_in, batch_first=True, dropout=0.0, norm_first=False)
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                edge_in = self._model_conf.edge_embed_size # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(node_embed_size=self._ipa_conf.c_s, edge_embed_in=edge_in,
                                                                                edge_embed_out=self._model_conf.edge_embed_size)


    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None, :] * node_mask[:, :, None]
        framework_mask = input_feats['framework_mask']
        diffuse_mask = input_feats['diffuse_mask']
        res_idx = input_feats['residue_index']
        chain_idx = input_feats['chain_index']
        ts_ = input_feats['t']
        trans_1 = input_feats['trans_1']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        aatype_idx = None
        chain_idx_oh = None
        hotspot_1d = None
        hotspot_2d = None

        if self._model_conf.node_features.embed_aa:
            aatype_idx = input_feats['aatype_oh']
        if self._model_conf.node_features.embed_chain:
            chain_idx_oh = input_feats['chain_idx_oh']
        if self._model_conf.node_features.use_1d_hotspot:
            hotspot_1d = input_feats['1d_hotspot'][..., None]
        if self._model_conf.edge_features.use_2d_hotspot:
            hotspot_2d = input_feats['2d_hotspot'][..., None]

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(ts_, node_mask, res_idx, chain_idx_oh, aatype_idx, hotspot_1d)
        
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']

        is_same_chain = (chain_idx[:, :, None]) == (chain_idx[:, None, :])
        is_same_entity = (diffuse_mask[:, :, None] == diffuse_mask[:, None, :])
        if self._model_conf.edge_features.same_diff_disto:
            dist_feats = calc_distogram(trans_1, min_bin=1e-3, max_bin=20.0, num_bins=self._model_conf.edge_features.num_bins)
            disto_cond_mask = (is_same_entity * (framework_mask[:, :, None] * framework_mask[:, None, :])).unsqueeze(-1)
            disto_uncond = torch.zeros_like(dist_feats)
            disto_uncond[..., -1] = 1
            dist_feats = dist_feats * disto_cond_mask + disto_uncond * (1-disto_cond_mask)
        else:
            dist_feats = calc_distogram(trans_t, min_bin=1e-3, max_bin=20.0, num_bins=self._model_conf.edge_features.num_bins)
            disto_cond_mask = None

        sc_feats = calc_distogram(trans_sc, min_bin=1e-3, max_bin=20.0, num_bins=self._model_conf.edge_features.num_bins)
        init_edge_embed = self.edge_feature_net(init_node_embed, dist_feats, sc_feats, edge_mask, 
                                                res_idx, node_mask, hotspot_2d, is_same_chain, disto_cond_mask)

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            # ipa_embed = self.trunk[f'ipa_{b}'](node_embed, torch.concat([edge_embed, hotspot_2d], -1), curr_rigids, node_mask)
            ipa_embed = self.trunk[f'ipa_{b}'](node_embed, edge_embed, curr_rigids, node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, (node_mask * diffuse_mask)[..., None])
            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        
        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        if self._model_conf.pred_aatype:
            aa_logits = self.aa_pred_head(node_embed)
            return {'pred_trans': pred_trans, 'pred_rotmats': pred_rotmats, 'aa_logits': aa_logits}
        else:
            return {'pred_trans': pred_trans, 'pred_rotmats': pred_rotmats}
