"""
Patch representation module using pose-normalized depth-ray grid.

Implements Option B1 from the plan: pose-normalized 3D points in a canonical patch frame,
with scale-invariant normalization for non-metric depth.
"""

import numpy as np
import torch
from .pose_utils import (
    normalize_rotation_matrix,
    extract_rotation_translation,
    compose_transform,
    invert_transform
)


def compute_patch_frame(
    center_world,
    camera_pose_wc,
    gravity_direction=None,
    local_points_world=None,
    use_gravity=True
):
    """
    Compute a canonical patch frame T_wp (world-from-patch) at the patch center.

    Uses hybrid approach:
    - z_p: gravity/world-z (if available and stable) or local plane normal
    - x_p: camera forward projected onto tangent plane
    - y_p: z_p × x_p

    Args:
        center_world: (3,) patch center in world coordinates
        camera_pose_wc: (4, 4) camera-to-world transform
        gravity_direction: (3,) optional gravity/world-up direction (default: [0,0,1])
        local_points_world: (N, 3) optional local 3D points for plane fitting fallback
        use_gravity: bool, whether to use gravity for z-axis (default: True)

    Returns:
        T_wp: (4, 4) world-from-patch transform
        confidence: float, confidence score for the patch frame (0-1)
    """
    is_torch = isinstance(center_world, torch.Tensor)

    if gravity_direction is None:
        if is_torch:
            gravity_direction = torch.tensor([0.0, 0.0, 1.0], dtype=center_world.dtype, device=center_world.device)
        else:
            gravity_direction = np.array([0.0, 0.0, 1.0])

    # Normalize gravity direction
    if is_torch:
        gravity_direction = gravity_direction / torch.norm(gravity_direction)
    else:
        gravity_direction = gravity_direction / np.linalg.norm(gravity_direction)

    # Extract camera forward direction (3rd column of R)
    R_cam, t_cam = extract_rotation_translation(camera_pose_wc)
    if is_torch:
        camera_forward = R_cam[:, 2]  # z-axis in camera frame (forward)
    else:
        camera_forward = R_cam[:, 2]

    # Choose z-axis for patch frame
    confidence = 1.0
    if use_gravity:
        z_p = gravity_direction
    elif local_points_world is not None and len(local_points_world) >= 3:
        # Fallback: fit plane to local points
        z_p, plane_confidence = fit_plane_normal(local_points_world, camera_forward)
        confidence = plane_confidence
    else:
        # Final fallback: use gravity
        z_p = gravity_direction

    # Normalize z_p
    if is_torch:
        z_p = z_p / torch.norm(z_p)
    else:
        z_p = z_p / np.linalg.norm(z_p)

    # Compute x-axis: project camera forward onto tangent plane
    if is_torch:
        fwd_projected = camera_forward - (camera_forward @ z_p) * z_p
        fwd_norm = torch.norm(fwd_projected)
    else:
        fwd_projected = camera_forward - (camera_forward @ z_p) * z_p
        fwd_norm = np.linalg.norm(fwd_projected)

    if fwd_norm < 1e-6:
        # Camera is looking up/down, use arbitrary tangent vector
        if is_torch:
            x_p = torch.tensor([1.0, 0.0, 0.0], dtype=z_p.dtype, device=z_p.device)
            x_p = x_p - (x_p @ z_p) * z_p
            x_p = x_p / torch.norm(x_p)
        else:
            x_p = np.array([1.0, 0.0, 0.0])
            x_p = x_p - (x_p @ z_p) * z_p
            x_p = x_p / np.linalg.norm(x_p)
    else:
        x_p = fwd_projected / fwd_norm

    # Compute y-axis: z × x
    if is_torch:
        y_p = torch.cross(z_p, x_p)
        y_p = y_p / torch.norm(y_p)
    else:
        y_p = np.cross(z_p, x_p)
        y_p = y_p / np.linalg.norm(y_p)

    # Construct rotation matrix R_wp (columns are axes in world frame)
    if is_torch:
        R_wp = torch.stack([x_p, y_p, z_p], dim=1)  # (3, 3)
    else:
        R_wp = np.stack([x_p, y_p, z_p], axis=1)

    # Normalize rotation to ensure det=+1
    R_wp = normalize_rotation_matrix(R_wp)

    # Construct T_wp
    T_wp = compose_transform(R_wp, center_world)

    return T_wp, confidence


def fit_plane_normal(points, camera_forward, min_inliers=10):
    """
    Fit a plane to a set of 3D points and return the normal.

    Uses SVD-based plane fitting with sign disambiguation.

    Args:
        points: (N, 3) 3D points
        camera_forward: (3,) camera forward direction for sign disambiguation
        min_inliers: minimum number of points required

    Returns:
        normal: (3,) plane normal pointing toward camera
        confidence: float, fit quality (based on singular value ratio)
    """
    is_torch = isinstance(points, torch.Tensor)

    if len(points) < min_inliers:
        # Not enough points
        if is_torch:
            return camera_forward / torch.norm(camera_forward), 0.0
        else:
            return camera_forward / np.linalg.norm(camera_forward), 0.0

    # Center points
    if is_torch:
        centroid = points.mean(dim=0)
        centered = points - centroid
        U, S, Vt = torch.linalg.svd(centered)
        normal = Vt[2, :]  # Last right singular vector
    else:
        centroid = points.mean(axis=0)
        centered = points - centroid
        U, S, Vt = np.linalg.svd(centered)
        normal = Vt[2, :]

    # Disambiguate normal direction (point toward camera)
    if is_torch:
        if torch.dot(normal, camera_forward) < 0:
            normal = -normal
        # Confidence: ratio of smallest to second-smallest singular value
        # High ratio = points are nearly coplanar
        confidence = float((S[2] / (S[1] + 1e-8)).item())
    else:
        if np.dot(normal, camera_forward) < 0:
            normal = -normal
        confidence = float(S[2] / (S[1] + 1e-8))

    # Invert confidence so low planar residual = high confidence
    confidence = 1.0 - np.clip(confidence, 0, 1)

    return normal, confidence


def backproject_depth_to_camera(depth_map, K, mask=None):
    """
    Backproject a depth map to 3D points in camera frame.

    Args:
        depth_map: (H, W) depth values
        K: (3, 3) intrinsic matrix
        mask: (H, W) optional validity mask

    Returns:
        points_cam: (H, W, 3) 3D points in camera frame
        valid_mask: (H, W) boolean mask of valid points
    """
    is_torch = isinstance(depth_map, torch.Tensor)

    H, W = depth_map.shape

    # Create pixel grid
    if is_torch:
        v, u = torch.meshgrid(torch.arange(H, device=depth_map.device),
                               torch.arange(W, device=depth_map.device),
                               indexing='ij')
        uv_homo = torch.stack([u, v, torch.ones_like(u)], dim=-1).float()  # (H, W, 3)

        # Invert K
        K_inv = torch.linalg.inv(K)

        # Backproject: X_c = D * K^-1 @ [u, v, 1]^T
        rays = torch.einsum('ij,hwj->hwi', K_inv, uv_homo)  # (H, W, 3)
        points_cam = rays * depth_map.unsqueeze(-1)  # (H, W, 3)

        # Validity mask
        valid_mask = depth_map > 0
        if mask is not None:
            valid_mask = valid_mask & mask

    else:
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        uv_homo = np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float32)

        K_inv = np.linalg.inv(K)
        rays = np.einsum('ij,hwj->hwi', K_inv, uv_homo)
        points_cam = rays * depth_map[..., None]

        valid_mask = depth_map > 0
        if mask is not None:
            valid_mask = valid_mask & mask

    return points_cam, valid_mask


def transform_points_to_patch_frame(points_world, T_wp):
    """
    Transform 3D points from world frame to patch frame.

    Args:
        points_world: (H, W, 3) or (N, 3) 3D points in world frame
        T_wp: (4, 4) world-from-patch transform

    Returns:
        points_patch: (H, W, 3) or (N, 3) 3D points in patch frame
    """
    is_torch = isinstance(points_world, torch.Tensor)

    # Invert T_wp to get T_pw (patch-from-world)
    T_pw = invert_transform(T_wp)

    R_pw, t_pw = extract_rotation_translation(T_pw)

    orig_shape = points_world.shape
    if points_world.ndim == 3:
        # (H, W, 3) -> (H*W, 3)
        points_flat = points_world.reshape(-1, 3)
    else:
        points_flat = points_world

    # Transform: X_p = R_pw @ X_w + t_pw
    if is_torch:
        points_patch_flat = torch.mm(points_flat, R_pw.T) + t_pw
    else:
        points_patch_flat = points_flat @ R_pw.T + t_pw

    if points_world.ndim == 3:
        points_patch = points_patch_flat.reshape(orig_shape)
    else:
        points_patch = points_patch_flat

    return points_patch


def compute_scale_normalization(points_patch, valid_mask, method='rms'):
    """
    Compute local scale for scale-invariant normalization.

    Args:
        points_patch: (H, W, 3) or (N, 3) 3D points in patch frame
        valid_mask: (H, W) or (N,) boolean mask of valid points
        method: 'rms' (RMS radius) or 'median' (median norm)

    Returns:
        scale: scalar scale factor
    """
    is_torch = isinstance(points_patch, torch.Tensor)

    # Flatten if needed
    if points_patch.ndim == 3:
        points_flat = points_patch[valid_mask]
    else:
        points_flat = points_patch[valid_mask] if valid_mask.ndim > 0 else points_patch

    if len(points_flat) == 0:
        return 1.0  # Fallback

    if is_torch:
        norms = torch.norm(points_flat, dim=-1)
        if method == 'rms':
            scale = torch.sqrt(torch.mean(norms ** 2))
        elif method == 'median':
            scale = torch.median(norms)
        else:
            raise ValueError(f"Unknown method: {method}")
        scale = scale.item()
    else:
        norms = np.linalg.norm(points_flat, axis=-1)
        if method == 'rms':
            scale = np.sqrt(np.mean(norms ** 2))
        elif method == 'median':
            scale = np.median(norms)
        else:
            raise ValueError(f"Unknown method: {method}")

    return max(scale, 1e-6)  # Prevent division by zero


def extract_patch_representation(
    rgb_crop,
    depth_crop,
    K,
    T_wc,
    patch_center_world,
    gravity_direction=None,
    include_rgb=False,
    include_relative_depth=True,
    scale_method='rms'
):
    """
    Extract pose-normalized, scale-invariant patch representation.

    This is the main function for creating patch tensors following Option B1.

    Args:
        rgb_crop: (3, H, W) or (H, W, 3) RGB image crop
        depth_crop: (H, W) depth map crop
        K: (3, 3) intrinsic matrix
        T_wc: (4, 4) camera-to-world transform
        patch_center_world: (3,) patch center in world coordinates
        gravity_direction: (3,) optional gravity direction
        include_rgb: bool, whether to include RGB features (default: False)
        include_relative_depth: bool, whether to include relative depth channel (default: True)
        scale_method: 'rms' or 'median' for scale normalization

    Returns:
        patch_tensor: (C, H, W) patch representation
            C = 5 if include_relative_depth else 4 (Xn, Yn, Zn, [z_rel], mask)
            C += 3 if include_rgb
        metadata: dict with scale, T_wp, etc.
    """
    is_torch = isinstance(depth_crop, torch.Tensor)

    H, W = depth_crop.shape

    # Backproject to camera frame
    points_cam, valid_mask = backproject_depth_to_camera(depth_crop, K)

    # Transform to world frame
    R_wc, t_wc = extract_rotation_translation(T_wc)
    if is_torch:
        points_world = torch.mm(points_cam.reshape(-1, 3), R_wc.T) + t_wc
        points_world = points_world.reshape(H, W, 3)
    else:
        points_world = points_cam.reshape(-1, 3) @ R_wc.T + t_wc
        points_world = points_world.reshape(H, W, 3)

    # Compute patch frame
    T_wp, frame_confidence = compute_patch_frame(
        patch_center_world,
        T_wc,
        gravity_direction=gravity_direction
    )

    # Transform to patch frame
    points_patch = transform_points_to_patch_frame(points_world, T_wp)

    # Compute scale
    scale = compute_scale_normalization(points_patch, valid_mask, method=scale_method)

    # Normalize by scale
    points_patch_norm = points_patch / scale

    # Compute patch center in patch frame (should be near origin)
    if is_torch:
        center_patch = transform_points_to_patch_frame(
            patch_center_world.unsqueeze(0),
            T_wp
        ).squeeze(0)
    else:
        center_patch = transform_points_to_patch_frame(
            patch_center_world[None],
            T_wp
        ).squeeze(0)

    # Recenter points (make patch center exactly at origin)
    points_patch_norm = points_patch_norm - center_patch

    # Build channels
    channels = []

    # XYZ normalized
    if is_torch:
        channels.append(points_patch_norm[..., 0])  # X
        channels.append(points_patch_norm[..., 1])  # Y
        channels.append(points_patch_norm[..., 2])  # Z
    else:
        channels.append(points_patch_norm[..., 0])
        channels.append(points_patch_norm[..., 1])
        channels.append(points_patch_norm[..., 2])

    # Relative depth channel (optional)
    if include_relative_depth:
        Z_patch = points_patch[..., 2]
        if is_torch:
            Z_median = torch.median(Z_patch[valid_mask])
            Z_mad = torch.median(torch.abs(Z_patch[valid_mask] - Z_median))
            z_rel = (Z_patch - Z_median) / (Z_mad + 1e-8)
        else:
            Z_median = np.median(Z_patch[valid_mask])
            Z_mad = np.median(np.abs(Z_patch[valid_mask] - Z_median))
            z_rel = (Z_patch - Z_median) / (Z_mad + 1e-8)
        channels.append(z_rel)

    # Mask channel
    if is_torch:
        channels.append(valid_mask.float())
    else:
        channels.append(valid_mask.astype(np.float32))

    # RGB features (optional)
    if include_rgb:
        # Ensure RGB is (H, W, 3)
        if rgb_crop.shape[0] == 3:
            if is_torch:
                rgb_crop = rgb_crop.permute(1, 2, 0)
            else:
                rgb_crop = rgb_crop.transpose(1, 2, 0)

        # Normalize RGB to [0, 1]
        if is_torch:
            rgb_normalized = rgb_crop.float() / 255.0 if rgb_crop.max() > 1 else rgb_crop
            channels.append(rgb_normalized[..., 0])  # R
            channels.append(rgb_normalized[..., 1])  # G
            channels.append(rgb_normalized[..., 2])  # B
        else:
            rgb_normalized = rgb_crop.astype(np.float32) / 255.0 if rgb_crop.max() > 1 else rgb_crop
            channels.append(rgb_normalized[..., 0])
            channels.append(rgb_normalized[..., 1])
            channels.append(rgb_normalized[..., 2])

    # Stack channels: (C, H, W)
    if is_torch:
        patch_tensor = torch.stack(channels, dim=0)
    else:
        patch_tensor = np.stack(channels, axis=0)

    # Metadata
    metadata = {
        'scale': scale,
        'T_wp': T_wp,
        'frame_confidence': frame_confidence,
        'valid_pixel_ratio': float(valid_mask.sum()) / (H * W),
        'patch_center_world': patch_center_world,
        'patch_center_patch': center_patch
    }

    return patch_tensor, metadata


def test_patch_representation():
    """
    Unit test for patch representation extraction.
    """
    print("Testing patch representation...")

    # Create synthetic data
    H, W = 64, 64
    depth = np.random.rand(H, W) * 5 + 1  # Depth in [1, 6]
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    K = np.array([
        [500, 0, W/2],
        [0, 500, H/2],
        [0, 0, 1]
    ])

    T_wc = np.eye(4)
    T_wc[:3, 3] = [0, 0, 2]  # Camera at (0, 0, 2)

    patch_center = np.array([0.5, 0.5, 1.5])

    # Extract patch
    patch_tensor, metadata = extract_patch_representation(
        rgb, depth, K, T_wc, patch_center,
        include_rgb=False,
        include_relative_depth=True
    )

    print(f"  Patch tensor shape: {patch_tensor.shape}")
    print(f"  Expected: (5, {H}, {W})")
    assert patch_tensor.shape == (5, H, W), "Patch tensor shape mismatch!"

    print(f"  Scale: {metadata['scale']:.4f}")
    print(f"  Valid pixel ratio: {metadata['valid_pixel_ratio']:.4f}")
    print(f"  Frame confidence: {metadata['frame_confidence']:.4f}")

    assert metadata['scale'] > 0, "Scale should be positive!"
    assert 0 <= metadata['valid_pixel_ratio'] <= 1, "Valid ratio should be in [0, 1]!"

    print("✓ Patch representation test passed!\n")


if __name__ == "__main__":
    test_patch_representation()
