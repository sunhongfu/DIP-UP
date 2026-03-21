"""
Pure Python inference pipeline for DIP-UP.

Deep Image Prior for MRI Phase Unwrapping (PhaseNet3D).
The network is jointly trained on the input at inference time (no general checkpoint).

Reference: Zhu X, et al. "Phase Unwrapping with Deep Image Prior." MRM 2021.
"""

import os
import sys
import math
import tempfile

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASENET_DIR = os.path.join(_HERE, "PhaseNet3D")

if _PHASENET_DIR not in sys.path:
    sys.path.insert(0, _PHASENET_DIR)

from Unet_1Chan_9Class import Unet_1Chan_9Class  # noqa: E402


def _tv_loss(x, mask):
    """Total variation loss within mask."""
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    loss = (
        (dx * mask[:, :, 1:, :, :]).abs().sum()
        + (dy * mask[:, :, :, 1:, :]).abs().sum()
        + (dz * mask[:, :, :, :, 1:]).abs().sum()
    )
    return loss


def _lap_loss(wrapped, unwrapped):
    """Laplacian consistency loss (unwrapped phase should share Laplacian with wrapped phase)."""
    diff = unwrapped - wrapped
    rounds = torch.round(diff / (2 * math.pi))
    residual = diff - rounds * 2 * math.pi
    return residual.abs().sum(), residual


def run_dipup(
    phase_nii_path: str,
    *,
    mask_nii_path: str | None = None,
    checkpoint_path: str | None = None,
    n_iter: int = 2000,
    lr: float = 1e-6,
    shift_base: int = 5,
    output_dir: str | None = None,
    progress_fn=None,
) -> str:
    """
    Run DIP-UP phase unwrapping in pure Python.

    Parameters
    ----------
    phase_nii_path : str    – wrapped phase NIfTI (3D, values in radians)
    mask_nii_path : str     – brain mask NIfTI (optional; defaults to non-zero voxels)
    checkpoint_path : str   – path to PhaseNet3D .pth file (random init if None)
    n_iter : int            – optimisation iterations (default 2000)
    lr : float              – RMSprop learning rate (default 1e-6)
    shift_base : int        – winding number shift base (default 5)
    output_dir : str        – output directory (temp dir if None)

    Returns
    -------
    unwrapped_path – path to the unwrapped phase NIfTI file
    """
    def _log(frac, msg):
        print(f"[{frac:.0%}] {msg}")
        if progress_fn:
            progress_fn(frac, msg)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="dipup_")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(0.0, f"Device: {device}")

    _log(0.05, "Loading phase …")
    phase_img = nib.load(phase_nii_path)
    affine = phase_img.affine
    phase_np = phase_img.get_fdata(dtype=np.float32)

    image = torch.from_numpy(phase_np).float().unsqueeze(0).unsqueeze(0).to(device)

    if mask_nii_path is not None:
        _log(0.08, "Loading mask …")
        mask_np = (nib.load(mask_nii_path).get_fdata() > 0.5).astype(np.float32)
        tissue_mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        tissue_mask = (image != 0).float()

    _log(0.10, "Building PhaseNet3D …")
    net = Unet_1Chan_9Class(4)
    net = nn.DataParallel(net)

    if checkpoint_path is not None:
        _log(0.12, f"Loading checkpoint from {checkpoint_path} …")
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))

    net = net.to(device)
    net.eval()

    opt = optim.RMSprop(net.parameters(), lr=lr)
    idx = torch.arange(0, 9, device=device)
    final_uwp = None

    _log(0.15, f"Optimising for {n_iter} iterations …")
    for it in range(n_iter):
        recons = net(image)
        recons_softmax = torch.softmax(recons, dim=1)
        idx2 = idx[None, :, None, None, None]
        recon_distri = idx2 * recons_softmax
        recon_count = recon_distri.sum(1).unsqueeze(1) - shift_base
        recon_count = recon_count * tissue_mask.squeeze(1).unsqueeze(1)
        recon_uwph = recon_count * 2 * math.pi + image

        loss1 = _tv_loss(recon_uwph, tissue_mask)
        loss2, _ = _lap_loss(image, recon_uwph)
        loss = loss1 + loss2

        loss.backward()
        opt.step()
        opt.zero_grad()

        if it % max(1, n_iter // 20) == 0:
            frac = 0.15 + 0.80 * it / n_iter
            _log(frac, f"Iter {it}/{n_iter} | TV={loss1.item():.3f} | Lap={loss2.item():.3f}")
            if it == n_iter - 1 or it % (n_iter // 5) == 0:
                with torch.no_grad():
                    recon_count_int = torch.round(recon_count)
                    final_uwp = (recon_count_int * 2 * math.pi + image).squeeze().cpu().numpy()

    if final_uwp is None:
        with torch.no_grad():
            recons = net(image)
            recons_softmax = torch.softmax(recons, dim=1)
            idx2 = idx[None, :, None, None, None]
            recon_count = (idx2 * recons_softmax).sum(1).unsqueeze(1) - shift_base
            recon_count = recon_count * tissue_mask.squeeze(1).unsqueeze(1)
            final_uwp = (torch.round(recon_count) * 2 * math.pi + image).squeeze().cpu().numpy()

    _log(0.95, "Saving …")
    unwrapped_path = os.path.join(output_dir, "DIP-UP_unwrapped.nii.gz")
    nib.save(nib.Nifti1Image(final_uwp.astype(np.float32), affine), unwrapped_path)

    _log(1.0, f"Done! Saved to {output_dir}")
    return unwrapped_path
