from collections import defaultdict
import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy
from torch import autograd
from motif_scaffolding import twisting
from tqdm import tqdm
import numpy as np


def _centered_gaussian(num_batch, num_res, device, seed=None):
    if seed:
        torch.manual_seed(seed)
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device, seed=None):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    noise = torch.tensor(Rotation.random(num_batch*num_res).as_matrix(), device=device).reshape(num_batch, num_res, 3, 3)
    return noise

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])
    
def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (1 - diffuse_mask[..., None, None])


class Interpolant:

    def __init__(self, cfg, use_cfg=True):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        self.use_cfg = use_cfg

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        num_batch, num_res, _ = trans_1.shape
        trans_nm_0 = _centered_gaussian(num_batch, num_res, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_batch*num_res).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (rotmats_t * res_mask[..., None, None] + identity[None, None] * (1 - res_mask[..., None, None]))
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)
        
        trans_1 = batch['trans_1']  # Angstrom , [B, N, 3]
        rotmats_1 = batch['rotmats_1'] # [B, N, 3, 3]
        res_mask = batch['res_mask'] # [B, N]
        num_batch, _ = res_mask.shape

        # [B, 1]
        ts_ = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = ts_

        # Apply corruptions
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(trans_1, ts_, res_mask, batch['diffuse_mask'])
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t

        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(rotmats_1, ts_, res_mask, batch['diffuse_mask'])
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')
        noisy_batch['rotmats_t'] = rotmats_t

        rigid_t = du.create_rigid(rotmats_t, trans_t)
        noisy_batch['rigid_t'] = rigid_t
        
        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_vector_field(self, t, trans_1, trans_t):
        return (trans_1 - trans_t) / (1 - t)

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t > 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

    def apply_cfg(self, pred_trans_1, neg_pred_trans_1, pred_rotmats_1, neg_pred_rotmats_1, gd_scale):
        pred_trans_1 = pred_trans_1 + gd_scale * (pred_trans_1 - neg_pred_trans_1)
        return pred_trans_1, pred_rotmats_1

    def sample(self, model, batch, num_timesteps=None, trans_potential=None, trans_0=None, rotmats_0=None, ts=None, seed=None):
        res_mask = batch['res_mask']
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch['diffuse_mask']
        motif_mask = ~diffuse_mask.bool().squeeze(0)
        trans_1 = batch['trans_1']
        rotmats_1 = batch['rotmats_1']
        

        # Set-up initial prior samples
        if 'trans_0' not in batch:
            trans_0 = _centered_gaussian(num_batch, num_res, self._device, seed) * du.NM_TO_ANG_SCALE
        if 'rotmats_0' not in batch:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device, seed)

        if not self._cfg.twisting.use: # amortisation
            diffuse_mask = diffuse_mask.expand(num_batch, -1) # shape = (B, num_residue)
            batch['diffuse_mask'] = diffuse_mask
            rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
            trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)
            if torch.isnan(trans_0).any():
                raise ValueError('NaN detected in trans_0')

        logs_traj = defaultdict(list)
        if self._cfg.twisting.use: # sampling / guidance
            assert trans_1.shape[0] == 1 # assume only one motif
            motif_locations = torch.nonzero(motif_mask).squeeze().tolist()
            true_motif_locations, motif_segments_length = twisting.find_ranges_and_lengths(motif_locations)

            # Marginalise both rotation and motif location
            assert len(motif_mask.shape) == 1
            trans_motif = trans_1[:, motif_mask]  # [1, motif_res, 3]
            R_motif = rotmats_1[:, motif_mask]  # [1, motif_res, 3, 3]
            num_res = trans_1.shape[-2]
            with torch.inference_mode(False):
                motif_locations = true_motif_locations if self._cfg.twisting.motif_loc else None
                F, motif_locations = twisting.motif_offsets_and_rots_vec_F(num_res, motif_segments_length, motif_locations=motif_locations, 
                                                                           num_rots=self._cfg.twisting.num_rots, align=self._cfg.twisting.align, 
                                                                           scale=self._cfg.twisting.scale_rots, trans_motif=trans_motif, 
                                                                           R_motif=R_motif, max_offsets=self._cfg.twisting.max_offsets, 
                                                                           device=self._device, return_rots=False)

        if motif_mask is not None and len(motif_mask.shape) == 1:
            motif_mask = motif_mask[None].expand((num_batch, -1))

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
            
        if ts is None:
            ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
            
        t_1 = ts[0]
        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []

        cfg_gd_sched = 1.0 * (1 + (torch.cos(torch.pi*ts)+1)/2) # cosine annealing
        cfg_gd_sched[int(len(ts)*0.5):] = 0.0 # early truncation

        with tqdm(total=len(ts[1:]), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as pbar:
            
            for i, t_2 in enumerate(ts[1:]):
                
                # Run model.
                trans_t_1, rotmats_t_1 = prot_traj[-1]
                if self._trans_cfg.corrupt:
                    batch['trans_t'] = trans_t_1
                else:
                    if trans_1 is None:
                        raise ValueError('Must provide trans_1 if not corrupting.')
                    batch['trans_t'] = trans_1
                if self._rots_cfg.corrupt:
                    batch['rotmats_t'] = rotmats_t_1
                else:
                    if rotmats_1 is None:
                        raise ValueError('Must provide rotmats_1 if not corrupting.')
                    batch['rotmats_t'] = rotmats_1
                batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
                d_t = t_2 - t_1
    
                use_twisting = self._cfg.twisting.use and t_1 >= self._cfg.twisting.t_min
    
                if use_twisting: # Reconstruction guidance
                    with torch.inference_mode(False):
                        batch, Log_delta_R, delta_x = twisting.perturbations_for_grad(batch)
                        model_out = model(batch)
                        t = batch['t']
                        trans_t_1, rotmats_t_1, logs_traj = self.guidance(trans_t_1, rotmats_t_1, model_out, motif_mask, R_motif, 
                                                                          trans_motif, Log_delta_R, delta_x, t, d_t, logs_traj)
                else:
                    with torch.no_grad():
                        model_out = model(batch)
    
                # Process model output.
                pred_trans_1 = model_out['pred_trans']
                pred_rotmats_1 = model_out['pred_rotmats']

                if self.use_cfg:
                    with torch.no_grad():
                        neg_batch = {}
                        for k,v in batch.items():
                            neg_batch[k] = v
                            if k in ['hotspot_1d', 'hotspot_2d']:
                                neg_batch[k] = torch.zeros_like(v)
                        neg_model_out = model(neg_batch)
                    neg_pred_trans_1 = neg_model_out['pred_trans']
                    neg_pred_rotmats_1 = neg_model_out['pred_rotmats']
                    if self.use_cfg:
                        pred_trans_1, pred_rotmats_1 = self.apply_cfg(pred_trans_1, neg_pred_trans_1, pred_rotmats_1, neg_pred_rotmats_1, cfg_gd_sched[i])

                clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
                batch['trans_sc'] = (pred_trans_1 * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None]))
    
                # Take reverse step
                trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
                rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
                
                if trans_potential is not None:
                    with torch.inference_mode(False):
                        grad_pred_trans_1 = pred_trans_1.clone().detach().requires_grad_(True)
                        pred_trans_potential = autograd.grad(outputs=trans_potential(grad_pred_trans_1), inputs=grad_pred_trans_1)[0]
                    if self._trans_cfg.potential_t_scaling:
                        trans_t_2 -= t_1 / (1 - t_1) * pred_trans_potential * d_t
                    else:
                        trans_t_2 -= pred_trans_potential * d_t
                if not self._cfg.twisting.use:
                    trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
                    rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)
    
                prot_traj.append((trans_t_2, rotmats_t_2))
                t_1 = t_2
                pbar.update(1)

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        if self._trans_cfg.corrupt:
            batch['trans_t'] = trans_t_1
        else:
            if trans_1 is None:
                raise ValueError('Must provide trans_1 if not corrupting.')
            batch['trans_t'] = trans_1
        if self._rots_cfg.corrupt:
            batch['rotmats_t'] = rotmats_t_1
        else:
            if rotmats_1 is None:
                raise ValueError('Must provide rotmats_1 if not corrupting.')
            batch['rotmats_t'] = rotmats_1
        batch['ts_'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        if self.use_cfg:
            neg_batch = {}
            for k,v in batch.items():
                neg_batch[k] = v
                if k in ['hotspot_1d', 'hotspot_2d']:
                    neg_batch[k] = torch.zeros_like(v)
            with torch.no_grad():
                neg_model_out = model(neg_batch)
            neg_pred_trans_1 = neg_model_out['pred_trans']
            neg_pred_rotmats_1 = neg_model_out['pred_rotmats']
            if self.use_cfg:
                pred_trans_1, pred_rotmats_1 = self.apply_cfg(pred_trans_1, neg_pred_trans_1, pred_rotmats_1, neg_pred_rotmats_1, cfg_gd_sched[i])

        clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        
        return atom37_traj, clean_atom37_traj, clean_traj
