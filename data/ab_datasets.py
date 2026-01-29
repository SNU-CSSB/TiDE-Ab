import numpy as np
import pandas as pd
import logging
import torch
import math
import os

from torch.utils.data import Dataset
from data import utils as du
from openfold.data import data_transforms
from openfold.utils.rigid_utils import Rotation, Rigid
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import special_ortho_group
from utils import experiments_utils as eu

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist


## define constants
hcdr1 = np.arange(26,32+1).tolist()
hcdr2 = np.arange(52,56+1).tolist()
hcdr3 =  np.arange(95,102+1).tolist()
lcdr1 = np.arange(24,34+1).tolist()
lcdr2 = np.arange(50,56+1).tolist()
lcdr3 = np.arange(89,97+1).tolist()
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

_rg_rots = torch.eye(3, dtype=torch.float64)
_rg_rots[0, 0] = -1
_rg_rots[2, 2] = -1


class AbDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset
        self.sampler_cfg = data_cfg.sampler

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self._train_dataset = AbDataset(dataset_cfg=self.dataset_cfg, split='train')
            self._val_dataset = AbDataset(dataset_cfg=self.dataset_cfg, split='val')
        if stage == "test":
            self._test_dataset = AbDataset(dataset_cfg=self.dataset_cfg, split='test')


    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._train_dataset,
            batch_sampler=LengthBatcher(
                sampler_cfg=self.sampler_cfg,
                metadata_csv=self._train_dataset.csv,
                rank=rank,
                num_replicas=num_replicas,
            ),
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            sampler=DistributedSampler(self._val_dataset, shuffle=False),
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )
    
    def test_dataloader(self):
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(self._test_dataset, shuffle=False)
        else:
            sampler = None 
            
        return DataLoader(
            self._test_dataset,
            sampler=sampler,
            shuffle=False if sampler is None else None, # sampler가 있으면 shuffle을 직접 줄 수 없음
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )


class AbDataset(Dataset):

    def __init__(self, *, dataset_cfg, split):
        self._log = logging.getLogger(__name__)
        self.split = split
        self._dataset_cfg = dataset_cfg
        self.root_dir = self._dataset_cfg.root_dir
        self.csv = pd.read_csv(self._dataset_cfg.csv_path.format(split=split))
        self.csv = self.csv.reset_index()

    def __getitem__(self, row_idx):
        csv_row = self.csv.iloc[row_idx]
        file_path = os.path.join(self.root_dir, f'{csv_row.pdb_name}.pkl')
        processed_feats = du.read_pkl(file_path)
        all_atom_positions = torch.tensor(processed_feats['atom_positions']).double()
        if self.split == 'train' and self._dataset_cfg.rand_rot:
            rand_R = special_ortho_group.rvs(3)
            all_atom_positions = torch.einsum('ab,cda->cdb', torch.tensor(rand_R), all_atom_positions)

        chain_idx = processed_feats['chain_index'].astype(int)
        chain_idx_list = list(set(chain_idx))
        if '#' in csv_row.pdb_name:
            target_idx = chain_idx_list[1]
        else:
            target_idx = chain_idx_list[2]

        diffuse_mask = (chain_idx < target_idx)
        chain_idx = np.where(chain_idx >= target_idx, target_idx, chain_idx)
        framework_mask = np.logical_or(processed_feats['framework_mask'], ~diffuse_mask)

        _, counts = np.unique(chain_idx, return_counts=True)
        residue_index = np.concatenate([np.arange(1, c + 1) for c in counts])

        ###################################
        ####### define hotspot mask #######
        ###################################
        bb_positions = all_atom_positions[:,1,:]
        ab_len = np.sum(diffuse_mask)
        cond_ = torch.cdist(bb_positions[:ab_len], bb_positions[ab_len:]) < 10 ## cutoff = 10
        i, j = torch.where(cond_)
        epitope_ifaces = j + ab_len
        hotspot = torch.zeros_like(bb_positions[:, 0])
        epitope_ifaces = torch.unique(epitope_ifaces)

        if self.split == 'train' and self._dataset_cfg.hotspot_ratio_max <= 1:
            rand_ratio = np.random.uniform(self._dataset_cfg.hotspot_ratio_min, self._dataset_cfg.hotspot_ratio_max)
            rand_idx = torch.randperm(epitope_ifaces.size(0))
            epitope_ifaces = epitope_ifaces[rand_idx[:int(epitope_ifaces.size(0) * rand_ratio)]]
        hotspot[epitope_ifaces] = 1
        hotspot_mtx = hotspot[:, None] * hotspot[None, :]

        ###################################
        ###### center around epitope ######
        ###################################
        center_1 = bb_positions * hotspot[:, None]
        center_1 = torch.sum(center_1, dim=0) / (torch.sum(hotspot) + 1)
        if self.split == 'train':
            center_1 -= torch.randn(3) * 0.1

        all_atom_positions -= center_1[None, :]
        rigids_1 = Rigid.from_3_points(all_atom_positions[:,2,:], all_atom_positions[:,1,:], all_atom_positions[:,0,:])
        rigids_1 = rigids_1.compose(Rigid(Rotation(rot_mats=_rg_rots), None))
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()

        aatype_tensor = torch.tensor(processed_feats['aatype'])
        chain_idx = torch.tensor(chain_idx)
        
        feats = {
            'aatype': aatype_tensor.float(), 
            'aatype_oh': torch.nn.functional.one_hot(aatype_tensor.long(), num_classes=21).float(),
            'atom_pos': all_atom_positions[...,:14,:].float(),
            'atom_mask': torch.tensor(processed_feats['atom_mask'][...,:14]),
            'rotmats_1': rotmats_1, 
            'trans_1': trans_1,
            'framework_mask': torch.tensor(framework_mask).float(), 
            'diffuse_mask': torch.tensor(diffuse_mask).float(),
            'res_mask': torch.ones_like(aatype_tensor).float(),
            'chain_index': chain_idx.float(), 
            'chain_idx_oh': torch.nn.functional.one_hot(chain_idx.long(), num_classes=3).float(),
            'residue_index': torch.tensor(residue_index).int(),
            'file_path': csv_row.pdb_name,
            'hotspot_1d': hotspot.float(),
            'hotspot_2d': hotspot_mtx.float(), 
                }

        return feats
    

    def __len__(self):
        return len(self.csv)


class LengthBatcher:
    def __init__(self, sampler_cfg, metadata_csv, num_device=1, seed=123, 
                 shuffle=True, num_replicas=None, rank=None):
        super().__init__()

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = num_replicas
        self.rank = rank
        self._sampler_cfg = sampler_cfg
        self.batch_size = self._sampler_cfg.batch_size
        self._data_csv = metadata_csv
        if 'cluster' in self._data_csv:
            total_items = self._data_csv['cluster'].nunique()
        else:
            total_items = len(self._data_csv)
        self._num_batches = math.ceil(total_items / (num_replicas * self.batch_size))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.sample_order = []

    def _sample_indices(self):
        if 'cluster' in self._data_csv:
            cluster_sample = self._data_csv.groupby('cluster').sample(
                1, random_state=self.seed + self.epoch + self.rank
            )
            indices = cluster_sample['index'].tolist()
        else:
            indices = self._data_csv['index'].tolist()
        return indices[self.rank::self.num_replicas]

    def _replica_epoch_batches(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch + self.rank)
        indices = self._sample_indices()
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = [indices[i] for i in new_order]
        replica_csv = self._data_csv.iloc[indices]
        sample_order = []

        for _, len_df in replica_csv.groupby('seq_len'):
            num_batches = math.ceil(len(len_df) / self.batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i*self.batch_size:(i+1)*self.batch_size]
                batch_indices = batch_df['index'].tolist()
                if not batch_indices: continue
                batch_repeats = max(1, self.batch_size // len(batch_indices))
                full_batch = (batch_indices * batch_repeats)[:self.batch_size]
                sample_order.append(full_batch)
        
        if self.shuffle:
            new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
            return [sample_order[i] for i in new_order]
        return sample_order

    def _create_batches(self):
        all_batches = []
        while len(all_batches) < self._num_batches:
            new_batches = self._replica_epoch_batches()
            if not new_batches: break
            all_batches.extend(new_batches)
        self.sample_order = all_batches[:self._num_batches]

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return self._num_batches if not self.sample_order else len(self.sample_order)