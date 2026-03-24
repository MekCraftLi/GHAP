# semantic_utils.py
# Projects 3D gaussians into each training camera, samples the corresponding semantic
# mask from edge_masks/, and returns a boolean tensor marking gaussians that appear as
# foreground in >= hit_threshold cameras.
#
# Usage:
#   from semantic_utils import get_protected_gaussian_mask
#   fg_mask = get_protected_gaussian_mask(gaussians, scene.getTrainCameras(), dataset.source_path)

from pathlib import Path
import numpy as np
import torch
from PIL import Image


def get_protected_gaussian_mask(
    gaussians,
    cameras,
    dataset_path,
    hit_threshold: int = 2,
) -> torch.BoolTensor:
    """
    Returns a boolean tensor of shape (N,) on CUDA.
      True  => gaussian appeared in the white (foreground) region of >= hit_threshold masks.
      False => background / unseen — safe to pass to GMR compression.

    Args:
        gaussians:     GaussianModel with .get_xyz (N,3) on CUDA.
        cameras:       list of Camera objects (scene.getTrainCameras()).
        dataset_path:  dataset root path — edge_masks/ subfolder must exist there.
        hit_threshold: minimum foreground-camera hits required to protect a gaussian.
    """
    mask_dir = Path(dataset_path) / "edge_masks"
    if not mask_dir.exists():
        print(f"[semantic_utils] WARN: mask directory not found: {mask_dir}. No gaussians protected.")
        N = gaussians.get_xyz.shape[0]
        return torch.zeros(N, dtype=torch.bool, device="cuda")

    xyz = gaussians.get_xyz.detach()              # (N, 3)
    N = xyz.shape[0]
    hit_counts = torch.zeros(N, dtype=torch.int32, device="cuda")

    ones = torch.ones(N, 1, device="cuda", dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)         # (N, 4), pre-built once

    n_cams = len(cameras)
    print(f"[semantic_utils] Projecting {N} gaussians across {n_cams} cameras...")

    for cam in cameras:
        mask_path = mask_dir / (cam.image_name + ".png")
        if not mask_path.exists():
            continue

        try:
            mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            mask = torch.from_numpy(mask_np).to("cuda")   # (H, W) uint8
        except Exception as e:
            print(f"[semantic_utils] Could not load {mask_path.name}: {e}")
            continue

        H, W = mask.shape

        # Project: xyz_h @ full_proj_transform -> clip coords (N, 4)
        # full_proj_transform is stored transposed in Camera.__init__ (col-major convention),
        # so xyz_h @ M correctly gives clip = [x_clip, y_clip, z_clip, w_clip].
        with torch.no_grad():
            clip = xyz_h @ cam.full_proj_transform  # (N, 4)

        w_clip = clip[:, 3]

        # Depth filter: discard points behind or very close to the camera
        depth_ok = w_clip > 0.2

        # Perspective divide (avoid /0 for culled points)
        safe_w = w_clip.clone()
        safe_w[~depth_ok] = 1.0
        ndc_x = clip[:, 0] / safe_w
        ndc_y = clip[:, 1] / safe_w

        # NDC [-1,1] -> pixel [0, W) and [0, H)
        # NDC y=+1 is the top of the image, so we flip.
        u = ((ndc_x + 1.0) * 0.5 * W).long()
        v = ((1.0 - ndc_y) * 0.5 * H).long()

        in_bounds = (
            depth_ok
            & (u >= 0) & (u < W)
            & (v >= 0) & (v < H)
        )

        u_valid = u[in_bounds]
        v_valid = v[in_bounds]
        is_fg = mask[v_valid, u_valid] > 127        # bool (M,)

        hit_counts[in_bounds] += is_fg.to(torch.int32)

        del clip, w_clip, safe_w, ndc_x, ndc_y, u, v, in_bounds
        del mask, u_valid, v_valid, is_fg
        torch.cuda.empty_cache()

    del ones, xyz_h
    torch.cuda.empty_cache()

    protected = hit_counts >= hit_threshold
    print(f"[semantic_utils] Protected (foreground) gaussians: {protected.sum().item()} / {N}")
    return protected
