import abc
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random
import json
import math
import mdtraj as md

from torch.utils.data import Dataset
from data import utils as du
from openfold.data import data_transforms
from openfold.utils.rigid_utils import Rotation, Rigid
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import special_ortho_group
from experiments import utils as eu

## define constants following the format of chothia file
hcdr1 = np.arange(26,32+1).tolist()
hcdr2 = np.arange(52,56+1).tolist()
hcdr3 =  np.arange(95,102+1).tolist()
lcdr1 = np.arange(24,34+1).tolist()
lcdr2 = np.arange(50,56+1).tolist()
lcdr3 = np.arange(89,97+1).tolist()
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


def ck_lp_mc(fname):
    if ':' in fname:
        c1, c2 = fname.split(':')
        if len(c1.split('_')[-1]) + len(c2.split('_')[-1]) > 2:
            return False
        else:
            return True
    else:
        return True

class AntibodyDataset(Dataset):

    def __init__(self, *, dataset_cfg, _split='train', no_evcls=92):
        self._log = logging.getLogger(__name__)
        self._split = _split
        self._dataset_cfg = dataset_cfg

        self._rg_rots = torch.eye(3, dtype=torch.float64)
        self._rg_rots[0, 0] = -1
        self._rg_rots[2, 2] = -1

        with open('./processed_sabdab2025/test_pdbs.txt', 'r') as f:
            test_pdbs = f.readlines()
            test_pdbs = [_.strip() for _ in test_pdbs]

        if self._split == 'test':
            self.raw_csv = pd.read_csv(self._dataset_cfg.csv_path)
            self.raw_csv = self.raw_csv[self.raw_csv.seq_len <= 1600]
            self.csv = self.raw_csv.sort_values('seq_len', ascending=False)
            self.csv['index'] = list(range(len(self.csv)))
        else:
            raw_csv_ = []
            for csv_path in self._dataset_cfg.csv_path:
                raw_csv_.append(pd.read_csv(csv_path))
            self.raw_csv = pd.concat(raw_csv_)

            self.raw_csv = self.raw_csv[self.raw_csv.num_chains >= self._dataset_cfg.filter.min_num_chains]
            self.raw_csv = self.raw_csv[self.raw_csv.seq_len <= self._dataset_cfg.filter.max_num_res]
            self.raw_csv = self.raw_csv[self.raw_csv.missing_ratio <= self._dataset_cfg.filter.max_missing_ratio]
            self.raw_csv = self.raw_csv[~self.raw_csv.pdb_name.isin(test_pdbs)]
            self.raw_csv = self.raw_csv[[ck_lp_mc(_) for _ in self.raw_csv.pdb_name]] ## temp
            self.raw_csv = self.raw_csv.sort_values('seq_len', ascending=False)

            ##########################
            ########## temp ##########
            clst_info_ = []
            for clst_path in self._dataset_cfg.cluster_path:
                clst_info_.append(pd.read_csv(clst_path))
            clst_info = pd.concat(clst_info_)

            self._pdb_to_cluster = {}
            for i in range(len(clst_info)):
                _ = clst_info.iloc[i]
                self._pdb_to_cluster[_.data_name] = int(_.cluster_id)

            self._missing_pdbs = 0
            try:
                self._max_cluster = max(_pdb_to_cluster.values())
            except:
                self._max_cluster = 0
            
            def cluster_lookup(pdb):
                pdb = pdb.replace('_updated', '')
                if pdb not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb] = self._max_cluster
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb]
            self.raw_csv['cluster'] = self.raw_csv['pdb_name'].map(cluster_lookup)

            ## split train / val dataset according to cluster id
            eval_clusters = [0,1,2,3,4,5,6,7,8,9]

            if self._split == 'train':
                self.csv = self.raw_csv[~self.raw_csv.cluster.isin(eval_clusters)]
                self._log.info(f'Training: {len(self.csv)} examples')
            elif self._split == 'val':
                self.csv = self.raw_csv[self.raw_csv.cluster.isin(eval_clusters)]
                self._log.info(f'Validation: {len(self.csv)} examples')
            ##########################
            ##########################
            
            self.csv['index'] = list(range(len(self.csv)))

    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        processed_feats = du.read_pkl(csv_row['processed_path'])

        if ':' in csv_row.pdb_name: ## loop-ppi

            if self._dataset_cfg.masking_method == 'ss': #['ss', 'dist', 'both']
                masking_method = 'ss'
            elif self._dataset_cfg.masking_method == 'dist':
                masking_method = 'dist'
            elif self._dataset_cfg.masking_method == 'both':
                if np.random.uniform(0,1) > 0.5:
                    masking_method = 'ss'
                else:
                    masking_method = 'dist'

            if masking_method == 'ss':
                traj = md.load(csv_row.raw_path) # MDtraj
                pdb_ss = md.compute_dssp(traj, simplified=True) # SS calculation
                pdb_ss_converted = list(''.join(pdb_ss[0]).replace('NA', '-').replace('CCCECCC', 'CCCCCCC').replace('CCCHCCC', 'CCCCCCC').replace('CCCEECCC', 'CCCCCCCC').replace('CCCHHCCC', 'CCCCCCCC').replace('CCCEEECCC', 'CCCCCCCCC').replace('CCCHHHCCC', 'CCCCCCCCC'))

            if np.random.uniform(0,1) > 0.5:
                first_chain_id = processed_feats['chain_index'][0]
                mix_idx_ = np.sum(processed_feats['chain_index'] == first_chain_id)
                for k in processed_feats.keys():
                    processed_feats[k] = np.concatenate([processed_feats[k][mix_idx_:, ...], processed_feats[k][:mix_idx_, ...]])
                processed_feats['chain_index'] = (processed_feats['chain_index'] == first_chain_id).astype(int)

                if masking_method == 'ss':
                    pdb_ss_converted = pdb_ss_converted[mix_idx_:] + pdb_ss_converted[:mix_idx_]

        all_atom_positions = torch.tensor(processed_feats['atom_positions']).double()
        if self._dataset_cfg.rand_rot and self._split == 'train':
            rand_R = special_ortho_group.rvs(3)
            all_atom_positions = torch.einsum('ab,cda->cdb', torch.tensor(rand_R), all_atom_positions)

        ####
        # Run through OpenFold data transforms.
        # chain_feats = {'aatype': torch.tensor(processed_feats['aatype']).long(),
        #                'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
        #                'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()}
        # chain_feats = data_transforms.atom37_to_frames(chain_feats)
        # rigids_1 = Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]

        ## define diffuse_mask & framework_mask
        chain_idx = processed_feats['chain_index'].astype(int)
        chain_idx_list = list(set(chain_idx))
        if len(chain_idx_list) == 2: 
            target_idx = chain_idx_list[1]
            inpaint_chains = chain_idx_list[:1]
            target_chains = chain_idx_list[1:]
        else:
            target_idx = chain_idx_list[2]
            inpaint_chains = chain_idx_list[:2]
            target_chains = chain_idx_list[2:]

        diffuse_mask = (chain_idx < target_idx)
        framework_mask = np.logical_or(processed_feats['framework_mask'], ~diffuse_mask)

        bb_positions = torch.tensor(all_atom_positions[:,1,:])
        ab_len = np.sum(diffuse_mask)
       
        if ':' in csv_row.pdb_name:
            cond = torch.cdist(bb_positions[:ab_len], bb_positions[ab_len:])
            cond_min = cond.min(dim=-1)[0]

            # print('    masking_method : ', masking_method)
            if masking_method == 'dist':
                rand_dist_thres = np.random.randint(11,13)
                loop_ranges = eu.ranges(torch.where(cond_min < rand_dist_thres)[0].numpy())
                # print('entire frag candidates : ', loop_ranges)

                frags_ = []
                for cand_ in loop_ranges:
                    left_, right_ = cand_
                    if right_-left_+1 < 4:
                        continue
                    tip_idx = np.random.randint(right_-left_+1)
                    rand_len = np.random.randint(4, min(20, right_-left_+1)+1)
                    if tip_idx-int(rand_len/2) < 0:
                        start_idx =left_
                        end_idx = left_+rand_len-1
                    elif tip_idx+int(rand_len/2) > right_-left_:
                        start_idx = right_-rand_len+1
                        end_idx = right_
                    else:
                        start_idx = left_+tip_idx-int(rand_len/2)
                        end_idx = left_+tip_idx+int(rand_len/2)-1
                    frags_.append((start_idx, end_idx))
                # print('after length filtering : ', frags_)

            elif masking_method == 'ss':
                cidx_change = list(np.where(processed_feats['chain_index'][1:] - processed_feats['chain_index'][:-1])[0] + 1) 
                pdb_ss_converted[:5] = ['-']*5
                pdb_ss_converted[-5:] = ['-']*5
                for cidx_change_ in cidx_change:
                    pdb_ss_converted[cidx_change_-5:cidx_change_+5] = ['-']*10

                loop_mask = (np.asarray(pdb_ss_converted) == 'C')
                loop_ranges = eu.ranges(torch.where(torch.tensor(diffuse_mask)*torch.tensor(loop_mask))[0].numpy())
                # print('entire frag candidates : ', loop_ranges)
                
                frags_ = []
                for cand_ in loop_ranges:
                    left_, right_ = cand_
                    if right_-left_+1 < 4:
                        continue
                    tip_idx = torch.argmin(cond_min[left_:right_+1]).item()
                    closest_dist = cond_min[left_+tip_idx]
                    if closest_dist > 10:
                        continue
                    rand_len = np.random.randint(4, min(20, right_-left_+1)+1)
                    if tip_idx-int(rand_len/2) < 0:
                        start_idx =left_
                        end_idx = left_+rand_len-1
                    elif tip_idx+int(rand_len/2) > right_-left_:
                        start_idx = right_-rand_len+1
                        end_idx = right_
                    else:
                        start_idx = left_+tip_idx-int(rand_len/2)
                        end_idx = left_+tip_idx+int(rand_len/2)-1
                    frags_.append((start_idx, end_idx))

            framework_mask = torch.ones_like(bb_positions[:, 0]).bool()
            if len(frags_) <= 3:
                rand_num_frags_ = 3
                for fidx_ in frags_:
                    framework_mask[fidx_[0]:fidx_[1]+1] = False
            else:
                random.shuffle(frags_)
                rand_num_frags_ = np.random.randint(3, min(6, len(frags_))+1)
                for fidx_ in frags_[:rand_num_frags_]:
                    framework_mask[fidx_[0]:fidx_[1]+1] = False
            # print('final set : ', frags_[:rand_num_frags_])

        cond_ = torch.cdist(bb_positions[:ab_len], bb_positions[ab_len:]) < 10 ## cutoff = 10
        i, j = torch.where(cond_)
        epitope_ifaces = j + ab_len
        epi_full = torch.zeros_like(bb_positions[:, 0])

        if self._dataset_cfg.hotspot_ratio_max <= 1 and self._split == 'train':
            rand_ratio = np.random.uniform(self._dataset_cfg.hotspot_ratio_min, self._dataset_cfg.hotspot_ratio_max)
            rand_idx = torch.randperm(epitope_ifaces.size(0))
            epitope_ifaces = epitope_ifaces[rand_idx[:int(epitope_ifaces.size(0) * rand_ratio)]]

        epi_full[epitope_ifaces] = 1
        hotspot_mtx = epi_full[:, None] * epi_full[None, :]   

        if self._split == 'train':
            center_1 = bb_positions * torch.tensor(~diffuse_mask)[:, None]
            center_1 = torch.sum(center_1, dim=0) / (np.sum(~diffuse_mask) + 1)
            center_1 -= torch.randn(3) * 0.1

        else:
            center_1 = bb_positions * torch.tensor(~diffuse_mask)[:, None]
            center_1 = torch.sum(center_1, dim=0) / (np.sum(~diffuse_mask) + 1)

        all_atom_positions -= center_1[None, :]
        rigids_1 = Rigid.from_3_points(all_atom_positions[:,2,:], all_atom_positions[:,1,:], all_atom_positions[:,0,:])
        rigids_1 = rigids_1.compose(Rigid(Rotation(rot_mats=self._rg_rots), None))
        ####
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()

        #########################################################

        feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(), 
            'atom_pos': torch.tensor(all_atom_positions[...,:14,:]),
            'atom_mask': torch.tensor(processed_feats['atom_mask'][...,:14]),
            'rotmats_1': rotmats_1, 
            'trans_1': trans_1,
            'framework_mask': torch.tensor(framework_mask), 
            'diffuse_mask': torch.tensor(diffuse_mask),
            'res_mask': torch.tensor(~processed_feats['missing']),
            'chain_index': torch.tensor(chain_idx).int(), 
            'residue_index': torch.tensor(processed_feats['residue_index']), 
            'inpaint_chains' : ':'.join([PDB_CHAIN_IDS[_] for _ in inpaint_chains]), 
            'target_chains' : ':'.join([PDB_CHAIN_IDS[_] for _ in target_chains]),
            'file_path': csv_row['processed_path'], 
            '1d_hotspot': epi_full,
            '2d_hotspot': hotspot_mtx, 
                }

        # #############################################
        # res_mask = torch.tensor(processed_feats['bb_mask'])
        
        # if 'chothia_residue_index' in processed_feats:
        #     chothia_residue_index = torch.tensor(processed_feats['chothia_residue_index'])
        # else:
        #     chothia_residue_index = ''

        # feats = {'aatype': chain_feats['aatype'], 
        #          'rotmats_1': rotmats_1, 'trans_1': trans_1,
        #         'framework_mask': torch.tensor(framework_mask), 'diffuse_mask': torch.tensor(diffuse_mask),
        #         'res_mask': res_mask, 'chain_idx': torch.tensor(chain_idx), 'res_idx': torch.tensor(res_idx), 
        #         'inpaint_chains' : ':'.join([PDB_CHAIN_IDS[_] for _ in inpaint_chains]), 
        #         'target_chains' : ':'.join([PDB_CHAIN_IDS[_] for _ in target_chains]),
        #         'file_path': csv_row['processed_path'], 'atom_mask': torch.tensor(processed_feats['atom_mask']),
        #         '1d_hotspot': epi_full, '2d_hotspot': hotspot_mtx, 'chothia_residue_index': chothia_residue_index}
        # #############################################
        
        return feats
    

    def __len__(self):
        return len(self.csv)


class LengthBatcher:

    def __init__(self, batch_size, metadata_csv, num_device=1, seed=123, shuffle=True):
        super().__init__()

        self.batch_size = batch_size
        self._data_csv = metadata_csv
        if 'cluster' in self._data_csv: # Each replica needs the same number of batches. We set the number of batches to arbitrarily be the number of examples per replica.
            num_batches = self._data_csv['cluster'].nunique()
        else:
            num_batches = len(self._data_csv)
        self._num_batches = (num_batches // num_device)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def _sample_indices(self):
        if 'cluster' in self._data_csv:
            cluster_sample = self._data_csv.groupby('cluster').sample(1, random_state=self.seed + self.epoch)
            return cluster_sample['index'].tolist()
        else:
            return self._data_csv['index'].tolist()
        
    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._sample_indices()
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = [indices[i] for i in new_order]
        replica_csv = self._data_csv.iloc[indices]
        
        sample_order = []
        for seq_len_, len_df in replica_csv.groupby('seq_len'):
            num_batches = math.ceil(len(len_df) / self.batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i*self.batch_size:(i+1)*self.batch_size]
                batch_indices = batch_df['index'].tolist()
                batch_repeats = math.floor(self.batch_size / len(batch_indices))
                sample_order.append(batch_indices * batch_repeats)
        
        # Remove any length bias.
        if self.shuffle:
            new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
            return [sample_order[i] for i in new_order]
        return sample_order

    def _create_batches(self):
        all_batches = []
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)