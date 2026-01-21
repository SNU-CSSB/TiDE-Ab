import os
import hydra
import numpy as np
import random
import pandas as pd
import py3Dmol
import pickle

from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import so3_utils
from data import utils as du
from data import all_atom
from data import protein
from data.interpolant import Interpolant
from data import residue_constants
from data.datasets import AntibodyDataset, LengthBatcher

from models.flow_model import FlowModel
from experiments import utils as eu
from analysis import metrics
from analysis import utils as au



##### set cfg values

with initialize(version_base=None, config_path="configs/"):
    cfg = compose(config_name="base")

cfg.antibody_dataset.task = 'inpainting'
cfg.antibody_dataset.rand_rot = True
cfg.antibody_dataset.center_epitope = True

cfg.model.node_features.embed_chain = True
cfg.model.node_features.embed_aa = True
cfg.model.edge_features.ori_rel_pos = False
cfg.model.use_1d_hotspot = True
cfg.model.use_2d_hotspot = True

cfg.experiment.num_devices = torch.cuda.device_count()
cfg.experiment.checkpointer.dirpath = 'output/sabdab/with_cemb_aaemb_hsps_inpaint_rdrt_epct/' ### need to modify
cfg.experiment.warm_start = '' ### need to modify ['', 'weights/pdb/published.ckpt']
cfg.experiment.training.aux_loss_weight = 0.5

cfg.interpolant.use_framework_mask = False
device = f'cuda:{torch.cuda.current_device()}'
interpolant = Interpolant(cfg.interpolant)
interpolant.set_device(device)

os.makedirs(cfg.experiment.checkpointer.dirpath, exist_ok=True)
summary_writer = SummaryWriter(log_dir=cfg.experiment.checkpointer.dirpath)



##### load datasets

cfg.antibody_dataset.filter.max_num_res = 840 # (A6000 : 650*2, 840*1) / (A5000 : 650*1) 
batch_size = 1

k = torch.eye(3).to(device)
_, vectors = torch.linalg.eigh(k)

train_dataset = AntibodyDataset(dataset_cfg=cfg.antibody_dataset, _split='train', no_evcls=20)
val_dataset = AntibodyDataset(dataset_cfg=cfg.antibody_dataset, _split='val', no_evcls=20)

train_dataloader = DataLoader(train_dataset,  batch_sampler=LengthBatcher(batch_size*cfg.experiment.num_devices,
                                                                          metadata_csv=train_dataset.csv))
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print('num of training data / clusters : ', len(train_dataset), train_dataset.csv.cluster.unique().shape[0])
print('num of validation data / clusters : ', len(val_dataset), val_dataset.csv.cluster.unique().shape[0])


##### load model

model = FlowModel(cfg.model)
model = torch.nn.DataParallel(model)
model = model.to(device)

############## if resume is True ##############
if 'latest.pt' in os.listdir(cfg.experiment.checkpointer.dirpath):
    print('Resume from the latest.')
    sdict = torch.load(os.path.join(cfg.experiment.checkpointer.dirpath, 'latest.pt'))

    steps = sdict['steps'] + 1
    prev_max = sdict['prev_max']
    
    cfg.experiment.trainer.min_epochs = sdict['epoch'] + 1
    model.load_state_dict(sdict['model'])

    optimizer = torch.optim.AdamW(params=model.parameters(), **cfg.experiment.optimizer)
    optimizer.load_state_dict(sdict['optimizer'])

elif cfg.experiment.warm_start:
    print('Warmstart training using the weights of pre-trained model.')
    sdict = torch.load(cfg.experiment.warm_start)

    prev_max = 9999
    new_state_dict = {}
    for n, v in sdict['state_dict'].items():
        name = n.replace("model.","")
        new_state_dict[name] = v
        
    pop_list = []
    cur_keys = model.module.state_dict().keys()
    for k in new_state_dict:
        if k not in cur_keys:
            print(k, '--> not in the current model')
            pop_list.append(k)
        elif new_state_dict[k].shape != model.module.state_dict()[k].shape:
            print(k, '--> size mismatch')
            pop_list.append(k)

    for k in pop_list:
        _ = new_state_dict.pop(k)
    model.load_state_dict(new_state_dict, strict=False)
    
    cfg.experiment.optimizer.lr *= 0.1
    optimizer = torch.optim.AdamW(params=model.parameters(), **cfg.experiment.optimizer)

else:
    print('Train from scratch.')
    steps = 0
    prev_max = 9999

    optimizer = torch.optim.AdamW(params=model.parameters(), **cfg.experiment.optimizer)

config_path = os.path.join(cfg.experiment.checkpointer.dirpath, 'config.yaml')
with open(config_path, 'w') as f:
    OmegaConf.save(config=cfg, f=f.name)



##### start training

training_cfg = cfg.experiment.training

for epoch in range(cfg.experiment.trainer.min_epochs, cfg.experiment.trainer.max_epochs):
    print('... training epoch ', epoch)

    ## train
    model = model.train()
    for batch in train_dataloader:
        noisy_batch = {}
        batch['aatype'][~batch['framework_mask']] = 20
        
        for k in batch:
            if k == 'file_path':
                continue
            elif k == 'chain_idx':
                noisy_batch[k] = batch[k].to(torch.float32).to(device)
                noisy_batch[k+'_oh'] = torch.nn.functional.one_hot(batch[k], num_classes=3).to(torch.float32).to(device)
            elif k == 'aatype':
                noisy_batch[k] = batch[k].to(torch.float32).to(device)
                noisy_batch[k+'_oh'] = torch.nn.functional.one_hot(batch[k], num_classes=21).to(torch.float32).to(device)
            else:
                noisy_batch[k] = batch[k].to(torch.float32).to(device)
        noisy_batch = interpolant.corrupt_batch(noisy_batch)

        if cfg.interpolant.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = model(noisy_batch)
                noisy_batch['trans_sc'] = (model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None] 
                                           + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None]))
        
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape
    
        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))
        if torch.any(torch.isnan(gt_rot_vf)):
            raise ValueError('NaN encountered in gt_rot_vf')
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 
        
        # Timestep used for normalization.
        ts_ = noisy_batch['ts_']
        ts_norm_scale = 1 - torch.min(ts_[..., None], torch.tensor(training_cfg.t_normalize_clip))
        ts_norm_scale = torch.pow(ts_norm_scale, 2)

        # Model output predictions.
        model_output = model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError('NaN encountered in pred_rots_vf')
    
        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / ts_norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / ts_norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum((gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None], dim=(-1, -2, -3)) / loss_denom
        
        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / ts_norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(trans_error ** 2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        trans_loss = torch.clamp(trans_loss, max=5)
        
        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / ts_norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(rots_vf_error ** 2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        
        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)
        
        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])
        
        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]
        
        dist_mat_loss = torch.sum((gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask, dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)
        
        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss * training_cfg.aux_loss_use_bb_loss + dist_mat_loss * training_cfg.aux_loss_use_pair_loss)
        auxiliary_loss *= (ts_[:, 0] > training_cfg.aux_loss_t_pass)
        auxiliary_loss *= cfg.experiment.training.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)
        
        se3_vf_loss += auxiliary_loss
        if torch.any(torch.isnan(se3_vf_loss)):
            raise ValueError('NaN loss encountered')
        batch_losses = {"trans_loss": trans_loss, "rots_vf_loss": rots_vf_loss, "auxiliary_loss": auxiliary_loss}
        
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {k: torch.mean(v) for k,v in batch_losses.items()}        
    
        # update parameters 
        optimizer.zero_grad()
        loss = 0
        for k in total_losses:
            loss += total_losses[k]
        loss.backward()
        optimizer.step()
    
        if steps % 500 == 0:
            for k in total_losses.keys():
                summary_writer.add_scalar(f"train/{k}", total_losses[k].item(), steps)
        steps += 1
        
    torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                'prev_max': prev_max, 'steps': steps}, os.path.join(cfg.experiment.checkpointer.dirpath, 'latest.pt'))
    
    ###############

    model = model.eval()
    print('... validation epoch ', epoch)
    val_epoch_metrics = {"trans_loss": [], "rots_vf_loss": [], "auxiliary_loss": [], "se3_vf_loss": []}
    val_idx = 0
    with torch.no_grad():
        for batch in val_dataloader:

            noisy_batch = {}
            batch['aatype'][~batch['framework_mask']] = 20
            batch['chain_idx'][batch['chain_idx'] >= 2] = 2
            
            for k in batch:
                if k == 'file_path':
                    continue
                elif k == 'chain_idx':
                    noisy_batch[k] = batch[k].to(torch.float32).to(device)
                    noisy_batch[k+'_oh'] = torch.nn.functional.one_hot(batch[k], num_classes=3).to(torch.float32).to(device)
                elif k == 'aatype':
                    noisy_batch[k] = batch[k].to(torch.float32).to(device)
                    noisy_batch[k+'_oh'] = torch.nn.functional.one_hot(batch[k], num_classes=21).to(torch.float32).to(device)
                else:
                    noisy_batch[k] = batch[k].to(torch.float32).to(device)
            noisy_batch = interpolant.corrupt_batch(noisy_batch)

            loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
            if torch.any(torch.sum(loss_mask, dim=-1) < 1):
                raise ValueError('Empty batch encountered')
            num_batch, num_res = loss_mask.shape
        
            # Ground truth labels
            gt_trans_1 = noisy_batch['trans_1']
            gt_rotmats_1 = noisy_batch['rotmats_1']
            rotmats_t = noisy_batch['rotmats_t']
            gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))
            if torch.any(torch.isnan(gt_rot_vf)):
                raise ValueError('NaN encountered in gt_rot_vf')
            gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 
            
            # Timestep used for normalization.
            ts_ = noisy_batch['ts_']
            ts_norm_scale = 1 - torch.min(ts_[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
            # Model output predictions.
            model_output = model(noisy_batch)
            pred_trans_1 = model_output['pred_trans']
            pred_rotmats_1 = model_output['pred_rotmats']
            pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
            if torch.any(torch.isnan(pred_rots_vf)):
                raise ValueError('NaN encountered in pred_rots_vf')
        
            # Backbone atom loss
            pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
            gt_bb_atoms *= training_cfg.bb_atom_scale / ts_norm_scale[..., None]
            pred_bb_atoms *= training_cfg.bb_atom_scale / ts_norm_scale[..., None]
            loss_denom = torch.sum(loss_mask, dim=-1) * 3
            bb_atom_loss = torch.sum((gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None], dim=(-1, -2, -3)) / loss_denom
            
            # Translation VF loss
            trans_error = (gt_trans_1 - pred_trans_1) / ts_norm_scale * training_cfg.trans_scale
            trans_loss = training_cfg.translation_loss_weight * torch.sum(trans_error ** 2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
            trans_loss = torch.clamp(trans_loss, max=5)
            
            # Rotation VF loss
            rots_vf_error = (gt_rot_vf - pred_rots_vf) / ts_norm_scale
            rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(rots_vf_error ** 2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
            
            # Pairwise distance loss
            gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
            gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
            pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
            pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)
            
            flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
            flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
            flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
            flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])
            
            gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
            pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
            pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]
            
            dist_mat_loss = torch.sum((gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask, dim=(1, 2))
            dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)
            
            se3_vf_loss = trans_loss + rots_vf_loss
            auxiliary_loss = (bb_atom_loss * training_cfg.aux_loss_use_bb_loss + dist_mat_loss * training_cfg.aux_loss_use_pair_loss)
            auxiliary_loss *= (ts_[:, 0] > training_cfg.aux_loss_t_pass)
            auxiliary_loss *= cfg.experiment.training.aux_loss_weight
            auxiliary_loss = torch.clamp(auxiliary_loss, max=5)
            
            se3_vf_loss += auxiliary_loss
            if torch.any(torch.isnan(se3_vf_loss)):
                raise ValueError('NaN loss encountered')
            val_epoch_metrics["trans_loss"].append(trans_loss.cpu().item())
            val_epoch_metrics["rots_vf_loss"].append(rots_vf_loss.cpu().item())
            val_epoch_metrics["auxiliary_loss"].append(auxiliary_loss.cpu().item())
            val_epoch_metrics["se3_vf_loss"].append(se3_vf_loss.cpu().item())

        for metric_name, metric_vals in val_epoch_metrics.items():
            summary_writer.add_scalar(f'val_/{metric_name}', np.mean(metric_vals), steps)

        if prev_max > np.mean(val_epoch_metrics['se3_vf_loss']):
            prev_max = np.mean(val_epoch_metrics['se3_vf_loss'])
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                        'prev_max': prev_max, 'steps': steps}, os.path.join(cfg.experiment.checkpointer.dirpath, 'best.pt'))

