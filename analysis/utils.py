import numpy as np
import os
import re
from data import protein
from openfold.utils import rigid_utils
import torch


Rigid = rigid_utils.Rigid

def create_full_prot(atom37: np.ndarray, atom37_mask: np.ndarray, chain_index=None, aatype=None, residue_index=None, b_factors=None):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    n = atom37.shape[0]
    if residue_index is None:
        residue_index = np.arange(n)
    if chain_index is None:
        chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(atom_positions=atom37, atom_mask=atom37_mask, aatype=aatype, 
                           residue_index=residue_index, chain_index=chain_index, b_factors=b_factors)


def write_prot_to_pdb(prot_pos: np.ndarray, file_path: str, chain_index: np.ndarray=None, aatype: np.ndarray=None,
                      residue_index: np.ndarray=None, overwrite=False, no_indexing=True, b_factors=None):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
        
    if not no_indexing:
        save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_path = file_path
        
    with open(save_path, 'w') as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(pos37, atom37_mask, aatype=aatype, b_factors=b_factors)
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(prot_pos, atom37_mask, chain_index, aatype, residue_index, b_factors)
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        f.write('END')
    return save_path


def align_pdb(target_, mobile_):
    
    atom_exists_mask_ = np.ones_like(target_)
    atom_exists_mask_[:,3:] = 0

    target = (torch.from_numpy(target_).type(torch.double).unsqueeze(0)).to(torch.float32)
    mobile = (torch.from_numpy(mobile_).type(torch.double).unsqueeze(0)).to(torch.float32)
    atom_exists_mask = (torch.from_numpy(atom_exists_mask_[...,0]).type(torch.bool).unsqueeze(0))
    
    # Number of structures in the batch
    batch_size = mobile.shape[0]
    
    # if [B, Nres, Natom, 3], resize
    if mobile.dim() == 4:
        mobile = mobile.view(batch_size, -1, 3)
    if target.dim() == 4:
        target = target.view(batch_size, -1, 3)
    if atom_exists_mask is not None and atom_exists_mask.dim() == 3:
        atom_exists_mask = atom_exists_mask.view(batch_size, -1)
    
    # Number of atoms
    num_atoms = mobile.shape[1]
    
    # Apply masks if provided
    mobile = mobile.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
    target = target.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
    
    num_valid_atoms = atom_exists_mask.sum(dim=-1, keepdim=True)
    # Compute centroids for each batch
    centroid_mobile = mobile.sum(dim=-2, keepdim=True) / num_valid_atoms.unsqueeze(-1)
    centroid_target = target.sum(dim=-2, keepdim=True) / num_valid_atoms.unsqueeze(-1)
    
    # Handle potential division by zero if all atoms are invalid in a structure
    centroid_mobile[num_valid_atoms == 0] = 0
    centroid_target[num_valid_atoms == 0] = 0
    
    # Center structures by subtracting centroids
    centered_mobile = mobile - centroid_mobile
    centered_target = target - centroid_target
    
    centered_mobile = centered_mobile.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
    centered_target = centered_target.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
    
    # Compute covariance matrix for each batch
    covariance_matrix = torch.matmul(centered_mobile.transpose(1, 2), centered_target)
    
    # Singular Value Decomposition for each batch
    u, _, v = torch.svd(covariance_matrix)
    
    # Calculate rotation matrices for each batch
    rotation_matrix = torch.matmul(u, v.transpose(1, 2)).transpose(-2, -1).to(torch.float32)
    translation = torch.matmul(-centroid_mobile, rotation_matrix) + centroid_target
    
    # # Extract atom positions and convert to batched tensors
    # mobile_atom_tensor = (torch.from_numpy(mobile_).type(torch.float32).unsqueeze(0))
    # aligned_atom_tensor = translation + torch.einsum("...ij,...j", rotation_matrix, mobile_atom_tensor)

    return translation, rotation_matrix