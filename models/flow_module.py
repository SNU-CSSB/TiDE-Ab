from typing import Any
import torch
import time
import os
import random
# import wandb
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.flow_model import FlowModel
from models import utils as mu
from data.interpolant import Interpolant 
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from utils import experiments_utils as eu
from pytorch_lightning.loggers.wandb import WandbLogger

eps = 1e-6
bonding_constraints = [(1.329, 0.014) , (116.568, 1.995) , (121.352, 2.315)] # in degree
clash_tolerance = [0.5, 5.7, 2.8, 3.8]



class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._infer_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = self._infer_cfg.inference_dir
        self._inference_num_per_sample = self._infer_cfg.inference_num_per_sample


    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch: Any):

        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape
        device = loss_mask.device
        loss_denom = torch.sum(loss_mask, dim=-1) * 3

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 

        # Timestep used for normalization.
        ts_ = noisy_batch['t']
        ts_norm_scale = 1 - torch.min(ts_[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / ts_norm_scale * training_cfg.trans_scale
        mse_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        trans_loss = 5.0 * torch.tanh(mse_loss / 5.0)

        # Rotation VF loss
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        if torch.any(torch.isnan(gt_rot_vf)):
            raise ValueError('NaN encountered in gt_rot_vf')
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError('NaN encountered in pred_rots_vf')
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / ts_norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / ts_norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / ts_norm_scale[..., None]
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (
            bb_atom_loss * training_cfg.aux_loss_use_bb_loss
            + dist_mat_loss * training_cfg.aux_loss_use_pair_loss
        )
        auxiliary_loss *= (ts_[:, 0] > training_cfg.aux_loss_t_pass)
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        auxiliary_loss = torch.tanh(auxiliary_loss / 5.0)

        se3_vf_loss += auxiliary_loss
        if torch.any(torch.isnan(se3_vf_loss)):
            raise ValueError('NaN loss encountered')
        
        batch_losses = {
            "trans_loss": trans_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
                        }

        if self.current_epoch>training_cfg.start_steric_loss and training_cfg.clash_loss_weight > 0:
            ## btw_residue_class_loss
            dists = torch.sqrt(eps + torch.sum((pred_bb_atoms[..., :, None, :, None, :] - pred_bb_atoms[..., None, :, None, :, :])** 2,dim=-1,))
            dists_mask = torch.triu(torch.ones(dists.size()[:3]), diagonal=2)[..., None, None].repeat(1,1,1,3,3).to(device)
            dists_mask *= (ts_[...,None,None,None] > training_cfg.clash_loss_t_pass)
            
            # Compute the lower bound for the allowed distances. shape (N, N, 3, 3)
            atom_radius = (torch.tensor([[[1.55, 1.7, 1.7]]])).to(device) # van_der_waals_radius = {'C': 1.7, 'N': 1.55, 'O': 1.52}
            dists_lower_bound = (dists_mask * (atom_radius[..., :, None, :, None] + atom_radius[..., None, :, None, :])).to(device)
            
            # Compute the error. shape (N, N, 3, 3)
            dists_to_low_error = dists_mask * torch.nn.functional.relu(dists_lower_bound - dists - clash_tolerance[0]) 
            btw_residue_clash_loss = training_cfg.clash_loss_weight * torch.sum(dists_to_low_error, dim=(1,2,3,4))
            se3_vf_loss += btw_residue_clash_loss
            batch_losses["clash_loss"] = btw_residue_clash_loss

            ## btw_residue_bond_loss
            this_ca_pos = pred_bb_atoms[..., :-1, 1, :]
            this_c_pos = pred_bb_atoms[..., :-1, 2, :]
            next_n_pos = pred_bb_atoms[..., 1:, 0, :]
            next_ca_pos = pred_bb_atoms[..., 1:, 1, :]

            has_no_gap_mask = ((noisy_batch['residue_index'][..., 1:]) == noisy_batch['residue_index'][..., :-1] + 1).to(device)
            fls = torch.tensor([[False]]).repeat(noisy_batch['residue_index'].shape[0],1).to(device)
            has_no_gap_mask_f = torch.cat([fls, has_no_gap_mask], dim=-1)
            has_no_gap_mask_b = torch.cat([has_no_gap_mask, fls], dim=-1)
            has_no_gap_mask = (has_no_gap_mask_f * has_no_gap_mask_b)[..., :-1]
            has_no_gap_mask *= (ts_ > training_cfg.clash_loss_t_pass)
            
            # Compute loss for the C--N bond.
            c_n_bond_length = torch.sqrt(eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1))
            gt_length, gt_stddev = [torch.ones_like(c_n_bond_length) * v for v in bonding_constraints[0]]
            c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
            c_n_loss_per_residue = torch.nn.functional.relu(c_n_bond_length_error - gt_stddev * clash_tolerance[1])
            c_n_loss = torch.sum(has_no_gap_mask * c_n_loss_per_residue, dim=-1)
            
            # Compute loss for the angles.
            ca_c_bond_length = torch.sqrt(eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1))
            n_ca_bond_length = torch.sqrt(eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1))
            c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
            c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
            n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]
            
            gt_angle, gt_stddev = bonding_constraints[1]
            ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
            ca_c_n_cos_angle = torch.rad2deg(torch.arccos(torch.clamp(ca_c_n_cos_angle, -1, 1)))
            ca_c_n_cos_angle_error = torch.sqrt(eps + (ca_c_n_cos_angle - gt_angle) ** 2)
            ca_c_n_loss_per_residue = torch.nn.functional.relu(ca_c_n_cos_angle_error - gt_stddev * clash_tolerance[2])
            ca_c_n_loss = torch.sum(has_no_gap_mask * ca_c_n_loss_per_residue, dim=-1)
            
            gt_angle, gt_stddev = bonding_constraints[2]
            c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
            c_n_ca_cos_angle = torch.rad2deg(torch.arccos(torch.clamp(c_n_ca_cos_angle, -1, 1)))
            c_n_ca_cos_angle_error = torch.sqrt(eps + torch.square(c_n_ca_cos_angle - gt_angle))
            c_n_ca_loss_per_residue = torch.nn.functional.relu(c_n_ca_cos_angle_error - gt_stddev * clash_tolerance[3])
            c_n_ca_loss = torch.sum(has_no_gap_mask * c_n_ca_loss_per_residue, dim=-1)

            btw_residue_loss = training_cfg.clash_loss_weight * (c_n_loss + ca_c_n_loss + c_n_ca_loss)
            se3_vf_loss += btw_residue_loss
            batch_losses["bond_loss"] = btw_residue_loss

        batch_losses["se3_vf_loss"] = se3_vf_loss
        return batch_losses



    def validation_step(self, batch: Any, batch_idx: int):
        self.interpolant.set_device(batch['aatype'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        batch_losses = self.model_step(noisy_batch)
        
        num_batch = batch_losses['trans_loss'].shape[0]
        avg_losses = {f"val_{k}": torch.mean(v) for k, v in batch_losses.items()}
        
        val_loss = avg_losses.get('val_se3_vf_loss', avg_losses['val_trans_loss'])
        self._log_scalar("valid/loss", val_loss, on_step=False, on_epoch=True, batch_size=num_batch)

        self.validation_epoch_metrics.append(avg_losses)
        return val_loss


    def on_validation_epoch_end(self):
        if not self.validation_epoch_metrics:
            return

        keys = self.validation_epoch_metrics[0].keys()
        for key in keys:
            all_vals = [m[key] for m in self.validation_epoch_metrics]
            avg_val = torch.stack(all_vals).mean()
            
            self._log_scalar(
                f'valid/{key.replace("val_", "")}',
                avg_val,
                on_step=False,
                on_epoch=True,
                prog_bar=True
            )
            
        self.validation_epoch_metrics.clear()


    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if 'valid' in key or on_epoch:
            sync_dist = True
            rank_zero_only = False 

        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
            
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )


    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        self.interpolant.set_device(batch['aatype'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)

        if self._infer_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = (
                    model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None]
                    + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
                )

        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {k: torch.mean(v) for k,v in batch_losses.items()}
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        ts_ = torch.squeeze(noisy_batch['t'])
        self._log_scalar("train/t", np.mean(du.to_numpy(ts_)), prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            batch_t = ts_
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        train_loss = total_losses['se3_vf_loss']
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)
        return train_loss


    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )


    def predict_step(self, batch, batch_idx):

        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg) 
        interpolant.set_device(device)

        sample_name = batch['file_path'][0]
        sample_dir = os.path.join(self.inference_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        
        aatype = du.to_numpy(batch['aatype'].long())[0]
        chain_index = du.to_numpy(batch['chain_index'].long())[0]
        residue_index = du.to_numpy(batch['residue_index'].long())[0]
        diffuse_mask = du.to_numpy(batch['diffuse_mask'])[0]

        for i in range(self._inference_num_per_sample):
            atom37_traj, model_traj, _ = interpolant.sample(self.model, batch, seed=batch_idx+i)
            bb_traj = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))[0]
            _ = eu.save_traj(
                bb_traj[-1],
                bb_traj,
                np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                diffuse_mask=diffuse_mask,
                output_dir=sample_dir,
                sidx=i,
                aatype=aatype,
                chain_index=chain_index,
                residue_index=residue_index,
            )
