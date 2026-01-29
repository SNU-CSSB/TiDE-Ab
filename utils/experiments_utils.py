"""Utility functions for experiments."""
import logging
import torch
import os
import copy
import random
import GPUtil
import numpy as np
import pandas as pd
from analysis import utils as au
from data import utils as du
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from motif_scaffolding import save_motif_segments
from openfold.utils import rigid_utils as ru
from data.all_atom import compute_backbone
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB import PDBIO
from io import StringIO


def get_gt_atom37(input_):
    trans_1 = input_['trans_1'].unsqueeze(0)
    rotmats_1 = input_['rotmats_1'].unsqueeze(0)
    res_mask = input_['res_mask']
    rigids = du.create_rigid(rotmats_1, trans_1)
    gt_atom37 = compute_backbone(rigids, torch.zeros(trans_1.shape[0], trans_1.shape[1], 2, device=trans_1.device))[0]
    gt_atom37 = du.adjust_oxygen_pos(gt_atom37.to(trans_1.device)[0], res_mask).numpy()
    return gt_atom37

def align_coords(gt_pdb, pred_pdb, chains, frmask=None):
    parser = PDBParser(QUIET=True)
    gt_str = parser.get_structure("reference", StringIO(gt_pdb))
    pred_str = parser.get_structure("reference", StringIO(pred_pdb))
    
    ## align antibodies
    gt_atoms = []
    pred_atoms = []
    fridx = 0
    for chain_idx in chains:
        if frmask is not None:
            gt_atoms += [res['CA'] for i, res in enumerate(gt_str[0][chain_idx]) if 'CA' in res and frmask[fridx+i]]
            pred_atoms += [res['CA'] for i, res in enumerate(pred_str[0][chain_idx]) if 'CA' in res and frmask[fridx+i]] 
            fridx += len(gt_str[0][chain_idx])
        else:
            gt_atoms += [res['CA'] for res in gt_str[0][chain_idx] if 'CA' in res]
            pred_atoms += [res['CA'] for res in pred_str[0][chain_idx] if 'CA' in res]
    
    super_imposer = Superimposer()
    super_imposer.set_atoms(gt_atoms, pred_atoms)
    super_imposer.apply(pred_str.get_atoms())
    io = PDBIO()
    io.set_structure(pred_str)
    output = StringIO()
    io.save(output)
    pred_pdb = output.getvalue()
    return pred_pdb


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))
    
def get_input(data_, device, length_var = 0, embed_chain=True):

    data = copy.deepcopy(data_)
    data['aatype'][~data['framework_mask']] = 20
    if embed_chain:
        data['chain_idx'][data['chain_idx'] >= 2] = 2
    hcdr3_idx = ranges(torch.where(~data['framework_mask'])[0].numpy())[2][0]
    
    if length_var == 0:
        batch_ = {}
        for k in data:
            if k in ['rotmats_1', 'trans_1', 'framework_mask', 'diffuse_mask', 'res_mask', 'res_idx', 'atom_mask', '1d_hotspot', '2d_hotspot']:
                batch_[k] = data[k].to(torch.float32).unsqueeze(0).to(device)
            elif k == 'chain_idx':
                batch_[k] = data[k].to(torch.float32).unsqueeze(0).to(device)
                if embed_chain:
                    batch_[k+'_oh'] = torch.nn.functional.one_hot(data[k], num_classes=3).to(torch.float32).unsqueeze(0).to(device)
            elif k == 'aatype':
                batch_[k] = data[k].to(torch.float32).unsqueeze(0).to(device)
                batch_[k+'_oh'] = torch.nn.functional.one_hot(data[k], num_classes=21).to(torch.float32).unsqueeze(0).to(device)
            else:
                continue
        return batch_

    elif length_var > 0:
        batch_ = {}
        for k in data:
            if k in ['rotmats_1', 'trans_1', 'framework_mask', 'diffuse_mask', 'res_mask', 'res_idx', 'atom_mask', '1d_hotspot', '2d_hotspot']:
                batch_[k] = torch.cat([data[k][:hcdr3_idx], 
                                       data[k][hcdr3_idx:hcdr3_idx+length_var],
                                       data[k][hcdr3_idx:]]).to(torch.float32).unsqueeze(0).to(device)
            elif k == 'chain_idx':
                batch_[k] = torch.cat([data[k][:hcdr3_idx], 
                                       data[k][hcdr3_idx:hcdr3_idx+length_var],
                                       data[k][hcdr3_idx:]]).to(torch.float32).unsqueeze(0).to(device)
                if embed_chain:
                    batch_[k+'_oh'] = torch.nn.functional.one_hot(torch.cat([data[k][:hcdr3_idx], 
                                                                         data[k][hcdr3_idx:hcdr3_idx+length_var],
                                                                         data[k][hcdr3_idx:]]), num_classes=3).to(torch.float32).unsqueeze(0).to(device)
            elif k == 'aatype':
                batch_[k] = torch.cat([data[k][:hcdr3_idx], 
                                       data[k][hcdr3_idx:hcdr3_idx+length_var],
                                       data[k][hcdr3_idx:]]).to(torch.float32).unsqueeze(0).to(device)
                batch_[k+'_oh'] = torch.nn.functional.one_hot(torch.cat([data[k][:hcdr3_idx], 
                                       data[k][hcdr3_idx:hcdr3_idx+length_var],
                                       data[k][hcdr3_idx:]]), num_classes=21).to(torch.float32).unsqueeze(0).to(device)
            else:
                continue
        return batch_
    
    elif length_var < 0:
        batch_ = {}
        for k in data:
            if k in ['rotmats_1', 'trans_1', 'framework_mask', 'diffuse_mask', 'res_mask', 'res_idx', 'atom_mask', '1d_hotspot', '2d_hotspot']:
                batch_[k] = torch.cat([data[k][:hcdr3_idx], data[k][hcdr3_idx-length_var:]]).to(torch.float32).unsqueeze(0).to(device)
            elif k == 'chain_idx':
                batch_[k] = torch.cat([data[k][:hcdr3_idx], data[k][hcdr3_idx-length_var:]]).to(torch.float32).unsqueeze(0).to(device)
                if embed_chain:
                    batch_[k+'_oh'] = torch.nn.functional.one_hot(torch.cat([data[k][:hcdr3_idx], data[k][hcdr3_idx-length_var:]]),
                                                              num_classes=3).to(torch.float32).unsqueeze(0).to(device)
            elif k == 'aatype':
                batch_[k] = torch.cat([data[k][:hcdr3_idx], data[k][hcdr3_idx-length_var:]]).to(torch.float32).unsqueeze(0).to(device)
                batch_[k+'_oh'] = torch.nn.functional.one_hot(torch.cat([data[k][:hcdr3_idx], data[k][hcdr3_idx-length_var:]]),
                                                              num_classes=21).to(torch.float32).unsqueeze(0).to(device)
            else:
                continue
        return batch_


def get_sampled_mask(contigs, length, rng=None, num_tries=1000000):
    '''
    Parses contig and length argument to sample scaffolds and motifs.

    Taken from rosettafold codebase.
    '''
    length_compatible=False
    count = 0
    while length_compatible is False:
        inpaint_chains=0
        contig_list = contigs.strip().split()
        sampled_mask = []
        sampled_mask_length = 0
        #allow receptor chain to be last in contig string
        if all([i[0].isalpha() for i in contig_list[-1].split(",")]):
            contig_list[-1] = f'{contig_list[-1]},0'
        for con in contig_list:
            if (all([i[0].isalpha() for i in con.split(",")[:-1]]) and con.split(",")[-1] == '0'):
                #receptor chain
                sampled_mask.append(con)
            else:
                inpaint_chains += 1
                #chain to be inpainted. These are the only chains that count towards the length of the contig
                subcons = con.split(",")
                subcon_out = []
                for subcon in subcons:
                    if subcon[0].isalpha():
                        subcon_out.append(subcon)
                        if '-' in subcon:
                            sampled_mask_length += (int(subcon.split("-")[1])-int(subcon.split("-")[0][1:])+1)
                        else:
                            sampled_mask_length += 1

                    else:
                        if '-' in subcon:
                            if rng is not None:
                                length_inpaint = rng.integers(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            else:
                                length_inpaint=random.randint(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += length_inpaint
                        elif subcon == '0':
                            subcon_out.append('0')
                        else:
                            length_inpaint=int(subcon)
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += int(subcon)
                sampled_mask.append(','.join(subcon_out))
        #check length is compatible 
        if length is not None:
            if sampled_mask_length >= length[0] and sampled_mask_length < length[1]:
                length_compatible = True
        else:
            length_compatible = True
        count+=1
        if count == num_tries: #contig string incompatible with this length
            raise ValueError("Contig string incompatible with --length range")
    return sampled_mask, sampled_mask_length, inpaint_chains


def dataset_creation(dataset_class, cfg, task):
    train_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=True,
    ) 
    eval_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=False,
    ) 
    return train_dataset, eval_dataset


def get_available_device(num_device):
    return GPUtil.getAvailable(order='memory', limit = 8)[:num_device]


def save_traj(
        sample: np.ndarray,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: str,
        aatype = None,
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
        aatype: [T, N, 21] amino acid probability vector trajectory.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues and 0 for motif
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, 'sample.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj.pdb')
    x0_traj_path = os.path.join(output_dir, 'x0_traj.pdb')

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    sample_path = au.write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
    )
    prot_traj_path = au.write_prot_to_pdb(
        bb_prot_traj,
        prot_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
    )
    x0_traj_path = au.write_prot_to_pdb(
        x0_traj,
        x0_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype
    )
    return {
        'sample_path': sample_path,
        'traj_path': prot_traj_path,
        'x0_traj_path': x0_traj_path,
    }


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened
