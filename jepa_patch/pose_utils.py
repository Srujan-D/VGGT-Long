"""
Pose normalization and validation utilities.

Ensures that rotation matrices are valid (det=+1) and provides
tools for verifying transform conventions (T_wc vs T_cw).
"""

import numpy as np
import torch


def normalize_rotation_matrix(R):
    """
    Project a 3x3 matrix to the nearest valid rotation matrix with det=+1.

    Uses SVD decomposition to ensure orthogonality and correct determinant.

    Args:
        R: (3, 3) rotation matrix (numpy or torch)

    Returns:
        R_normalized: (3, 3) valid rotation matrix with det=+1
    """
    is_torch = isinstance(R, torch.Tensor)

    if is_torch:
        device = R.device
        dtype = R.dtype
        R_np = R.detach().cpu().numpy()
    else:
        R_np = R

    # SVD decomposition
    U, S, Vt = np.linalg.svd(R_np)

    # Reconstruct orthogonal matrix
    R_hat = U @ Vt

    # Ensure det = +1 (not -1)
    if np.linalg.det(R_hat) < 0:
        # Flip the last column of U
        U[:, 2] *= -1
        R_hat = U @ Vt

    if is_torch:
        R_hat = torch.from_numpy(R_hat).to(device=device, dtype=dtype)

    return R_hat


def normalize_rotation_matrix_batch(R_batch):
    """
    Normalize a batch of rotation matrices.

    Args:
        R_batch: (B, 3, 3) or (N, 3, 3) rotation matrices

    Returns:
        R_normalized: (B, 3, 3) valid rotation matrices
    """
    is_torch = isinstance(R_batch, torch.Tensor)

    if is_torch:
        device = R_batch.device
        dtype = R_batch.dtype
        R_batch = R_batch.detach().cpu().numpy()

    B = R_batch.shape[0]
    R_normalized = np.zeros_like(R_batch)

    for i in range(B):
        R_normalized[i] = normalize_rotation_matrix(R_batch[i])

    if is_torch:
        R_normalized = torch.from_numpy(R_normalized).to(device=device, dtype=dtype)

    return R_normalized


def extract_rotation_translation(T):
    """
    Extract rotation and translation from 4x4 homogeneous transform.

    Args:
        T: (4, 4) or (B, 4, 4) homogeneous transformation matrix

    Returns:
        R: (3, 3) or (B, 3, 3) rotation matrix
        t: (3,) or (B, 3) translation vector
    """
    if T.ndim == 2:
        R = T[:3, :3]
        t = T[:3, 3]
    else:
        R = T[:, :3, :3]
        t = T[:, :3, 3]
    return R, t


def compose_transform(R, t):
    """
    Compose 4x4 homogeneous transform from R and t.

    Args:
        R: (3, 3) or (B, 3, 3) rotation matrix
        t: (3,) or (B, 3) translation vector

    Returns:
        T: (4, 4) or (B, 4, 4) homogeneous transformation matrix
    """
    is_torch = isinstance(R, torch.Tensor)

    if R.ndim == 2:
        # Single transform
        if is_torch:
            T = torch.eye(4, dtype=R.dtype, device=R.device)
            T[:3, :3] = R
            T[:3, 3] = t
        else:
            T = np.eye(4, dtype=R.dtype)
            T[:3, :3] = R
            T[:3, 3] = t
    else:
        # Batch of transforms
        B = R.shape[0]
        if is_torch:
            T = torch.eye(4, dtype=R.dtype, device=R.device).unsqueeze(0).repeat(B, 1, 1)
            T[:, :3, :3] = R
            T[:, :3, 3] = t
        else:
            T = np.eye(4, dtype=R.dtype)[None].repeat(B, axis=0)
            T[:, :3, :3] = R
            T[:, :3, 3] = t

    return T


def invert_transform(T):
    """
    Invert homogeneous transformation matrix.

    For SE(3): T^-1 = [R^T | -R^T @ t]

    Args:
        T: (4, 4) or (B, 4, 4) transformation matrix

    Returns:
        T_inv: (4, 4) or (B, 4, 4) inverted transformation
    """
    R, t = extract_rotation_translation(T)

    is_torch = isinstance(T, torch.Tensor)

    if T.ndim == 2:
        R_inv = R.T
        t_inv = -R_inv @ t
    else:
        if is_torch:
            R_inv = R.transpose(-2, -1)
            t_inv = -torch.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)
        else:
            R_inv = np.transpose(R, (0, 2, 1))
            t_inv = -np.einsum('bij,bj->bi', R_inv, t)

    return compose_transform(R_inv, t_inv)


def verify_transform_convention(T_wc, K, depth, u, v, X_w):
    """
    Verify transform convention by reprojecting a 3D world point.

    This sanity check ensures that the pose convention (T_wc vs T_cw) is correct
    by checking if a known 3D world point reprojects correctly.

    Args:
        T_wc: (4, 4) world-from-camera transform (C2W)
        K: (3, 3) intrinsic matrix
        depth: scalar depth value at pixel (u, v)
        u, v: pixel coordinates (scalars)
        X_w: (3,) expected 3D world point

    Returns:
        reproj_error: scalar reprojection error (should be ~0)
    """
    is_torch = isinstance(T_wc, torch.Tensor)

    # Backproject pixel to camera space
    if is_torch:
        K_inv = torch.linalg.inv(K)
        pixel_homo = torch.tensor([u, v, 1.0], dtype=T_wc.dtype, device=T_wc.device)
        X_c = depth * (K_inv @ pixel_homo)
        X_c_homo = torch.cat([X_c, torch.ones(1, dtype=X_c.dtype, device=X_c.device)])
    else:
        K_inv = np.linalg.inv(K)
        pixel_homo = np.array([u, v, 1.0])
        X_c = depth * (K_inv @ pixel_homo)
        X_c_homo = np.concatenate([X_c, np.ones(1)])

    # Transform to world space
    X_w_pred = (T_wc @ X_c_homo)[:3]

    # Compute error
    if is_torch:
        error = torch.norm(X_w_pred - X_w).item()
    else:
        error = np.linalg.norm(X_w_pred - X_w)

    return error


def check_rotation_validity(R, tol=1e-5):
    """
    Check if a rotation matrix is valid.

    Criteria:
    - R^T @ R should be identity (orthogonality)
    - det(R) should be +1 (proper rotation, not reflection)

    Args:
        R: (3, 3) or (B, 3, 3) rotation matrix
        tol: tolerance for checks

    Returns:
        valid: bool or (B,) bool array indicating validity
    """
    is_torch = isinstance(R, torch.Tensor)

    if R.ndim == 2:
        # Single matrix
        if is_torch:
            identity = torch.eye(3, dtype=R.dtype, device=R.device)
            orthogonal = torch.allclose(R.T @ R, identity, atol=tol)
            det_check = torch.abs(torch.det(R) - 1.0) < tol
        else:
            identity = np.eye(3, dtype=R.dtype)
            orthogonal = np.allclose(R.T @ R, identity, atol=tol)
            det_check = np.abs(np.linalg.det(R) - 1.0) < tol

        valid = orthogonal and det_check
    else:
        # Batch
        B = R.shape[0]
        if is_torch:
            identity = torch.eye(3, dtype=R.dtype, device=R.device).unsqueeze(0).repeat(B, 1, 1)
            RtR = torch.bmm(R.transpose(-2, -1), R)
            orthogonal = torch.allclose(RtR, identity, atol=tol)
            dets = torch.det(R)
            det_check = torch.abs(dets - 1.0) < tol
        else:
            identity = np.eye(3, dtype=R.dtype)[None].repeat(B, axis=0)
            RtR = np.einsum('bij,bjk->bik', np.transpose(R, (0, 2, 1)), R)
            orthogonal = np.allclose(RtR, identity, atol=tol)
            dets = np.linalg.det(R)
            det_check = np.abs(dets - 1.0) < tol

        valid = orthogonal and det_check.all()

    return valid


def test_pose_normalization():
    """
    Unit test for pose normalization.
    """
    print("Testing pose normalization...")

    # Test 1: Valid rotation should remain unchanged
    R_valid = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    R_norm = normalize_rotation_matrix(R_valid)
    assert np.allclose(R_norm, R_valid), "Valid rotation was modified!"
    assert check_rotation_validity(R_norm), "Normalized rotation is invalid!"
    print("✓ Test 1 passed: Valid rotation unchanged")

    # Test 2: Slightly perturbed rotation should be corrected
    R_perturbed = R_valid + np.random.randn(3, 3) * 0.01
    R_norm = normalize_rotation_matrix(R_perturbed)
    assert check_rotation_validity(R_norm), "Normalized rotation is invalid!"
    assert np.abs(np.linalg.det(R_norm) - 1.0) < 1e-10, "Det is not +1!"
    print("✓ Test 2 passed: Perturbed rotation corrected")

    # Test 3: Reflection (det = -1) should be flipped
    R_reflect = np.diag([-1, 1, 1])
    assert np.linalg.det(R_reflect) < 0, "Should be a reflection"
    R_norm = normalize_rotation_matrix(R_reflect)
    assert np.linalg.det(R_norm) > 0, "Det should be positive after normalization!"
    assert check_rotation_validity(R_norm), "Normalized rotation is invalid!"
    print("✓ Test 3 passed: Reflection corrected")

    # Test 4: Batch normalization
    R_batch = np.array([R_valid, R_perturbed, R_reflect])
    R_batch_norm = normalize_rotation_matrix_batch(R_batch)
    for i in range(3):
        assert check_rotation_validity(R_batch_norm[i]), f"Batch[{i}] invalid!"
    print("✓ Test 4 passed: Batch normalization works")

    print("All pose normalization tests passed! ✓\n")


if __name__ == "__main__":
    test_pose_normalization()
