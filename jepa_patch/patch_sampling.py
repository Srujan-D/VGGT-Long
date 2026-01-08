"""
Patch sampling with geometric importance.

Samples patch centers from frames based on geometric complexity
to avoid planar/textureless regions and ensure informative training samples.
"""

import numpy as np
import torch
import cv2


def compute_depth_edges(depth_map, kernel_size=3):
    """
    Compute depth edge magnitude using Sobel operator.

    Args:
        depth_map: (H, W) depth values
        kernel_size: Sobel kernel size (default: 3)

    Returns:
        edge_magnitude: (H, W) edge magnitude
    """
    is_torch = isinstance(depth_map, torch.Tensor)

    if is_torch:
        depth_np = depth_map.detach().cpu().numpy()
    else:
        depth_np = depth_map

    # Replace invalid depth with 0
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Sobel gradients
    grad_x = cv2.Sobel(depth_np, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(depth_np, cv2.CV_64F, 0, 1, ksize=kernel_size)

    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    if is_torch:
        edge_magnitude = torch.from_numpy(edge_magnitude).to(
            device=depth_map.device,
            dtype=depth_map.dtype
        )

    return edge_magnitude


def compute_depth_variance(depth_map, window_size=7):
    """
    Compute local depth variance in a sliding window.

    Args:
        depth_map: (H, W) depth values
        window_size: window size for variance computation (default: 7)

    Returns:
        variance_map: (H, W) local depth variance
    """
    is_torch = isinstance(depth_map, torch.Tensor)

    if is_torch:
        depth_np = depth_map.detach().cpu().numpy()
    else:
        depth_np = depth_map

    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute mean
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)
    mean = cv2.filter2D(depth_np, cv2.CV_64F, kernel)

    # Compute variance: E[X^2] - E[X]^2
    mean_sq = cv2.filter2D(depth_np ** 2, cv2.CV_64F, kernel)
    variance = mean_sq - mean ** 2
    variance = np.maximum(variance, 0)  # Numerical stability

    if is_torch:
        variance = torch.from_numpy(variance).to(
            device=depth_map.device,
            dtype=depth_map.dtype
        )

    return variance


def compute_geometric_complexity_score(
    depth_map,
    alpha=1.0,
    beta=1.0,
    edge_kernel_size=3,
    var_window_size=7
):
    """
    Compute geometric complexity score for patch sampling.

    Score combines depth edges and local variance:
        score(u) = alpha * edge(u) + beta * variance(u)

    Higher scores indicate more geometric structure (edges, corners, texture).

    Args:
        depth_map: (H, W) depth values
        alpha: weight for edge magnitude (default: 1.0)
        beta: weight for depth variance (default: 1.0)
        edge_kernel_size: Sobel kernel size (default: 3)
        var_window_size: variance window size (default: 7)

    Returns:
        score_map: (H, W) complexity score
    """
    edges = compute_depth_edges(depth_map, kernel_size=edge_kernel_size)
    variance = compute_depth_variance(depth_map, window_size=var_window_size)

    # Normalize to [0, 1]
    is_torch = isinstance(depth_map, torch.Tensor)

    if is_torch:
        edges_norm = edges / (torch.max(edges) + 1e-8)
        variance_norm = variance / (torch.max(variance) + 1e-8)
    else:
        edges_norm = edges / (np.max(edges) + 1e-8)
        variance_norm = variance / (np.max(variance) + 1e-8)

    score = alpha * edges_norm + beta * variance_norm

    return score


def sample_patch_centers(
    depth_map,
    num_samples,
    crop_size=64,
    border_margin=32,
    alpha=1.0,
    beta=1.0,
    uniform_epsilon=0.05,
    depth_range=(0.1, 100.0),
    seed=None
):
    """
    Sample patch centers based on geometric complexity.

    Uses importance sampling with a mixture of:
    - Geometric complexity score (edges + variance)
    - Uniform distribution (epsilon mass)

    Args:
        depth_map: (H, W) depth values
        num_samples: number of patch centers to sample
        crop_size: size of patch crop (for border exclusion)
        border_margin: additional border margin to avoid edge artifacts
        alpha: weight for edge magnitude in score (default: 1.0)
        beta: weight for depth variance in score (default: 1.0)
        uniform_epsilon: mass for uniform sampling (default: 0.05)
        depth_range: (min, max) valid depth range (default: (0.1, 100.0))
        seed: random seed (optional)

    Returns:
        centers: (num_samples, 2) pixel coordinates (u, v)
        scores: (num_samples,) complexity scores at sampled locations
    """
    is_torch = isinstance(depth_map, torch.Tensor)

    if seed is not None:
        np.random.seed(seed)
        if is_torch:
            torch.manual_seed(seed)

    H, W = depth_map.shape
    half_crop = crop_size // 2

    # Compute complexity score
    score_map = compute_geometric_complexity_score(
        depth_map,
        alpha=alpha,
        beta=beta
    )

    # Create valid mask
    if is_torch:
        valid_mask = (
            (depth_map > depth_range[0]) &
            (depth_map < depth_range[1])
        )
    else:
        valid_mask = (
            (depth_map > depth_range[0]) &
            (depth_map < depth_range[1])
        )

    # Exclude borders
    valid_mask[:border_margin, :] = False
    valid_mask[-border_margin:, :] = False
    valid_mask[:, :border_margin] = False
    valid_mask[:, -border_margin:] = False

    # Exclude regions too close to image border for cropping
    valid_mask[:half_crop, :] = False
    valid_mask[-half_crop:, :] = False
    valid_mask[:, :half_crop] = False
    valid_mask[:, -half_crop:] = False

    # Get valid pixel coordinates
    if is_torch:
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # (N, 2) [v, u]
        if len(valid_indices) == 0:
            raise ValueError("No valid pixels found for sampling!")
        scores_valid = score_map[valid_mask]
    else:
        valid_indices = np.argwhere(valid_mask)  # (N, 2) [v, u]
        if len(valid_indices) == 0:
            raise ValueError("No valid pixels found for sampling!")
        scores_valid = score_map[valid_mask]

    # Create sampling distribution: (1 - epsilon) * scores + epsilon * uniform
    if is_torch:
        probs = (1 - uniform_epsilon) * scores_valid + uniform_epsilon
        probs = probs / torch.sum(probs)

        # Sample indices
        sample_indices = torch.multinomial(
            probs,
            num_samples=min(num_samples, len(probs)),
            replacement=False
        )

        centers = valid_indices[sample_indices]  # (num_samples, 2) [v, u]
        sampled_scores = scores_valid[sample_indices]

        # Convert to (u, v) format
        centers = torch.flip(centers, dims=[-1])  # [u, v]

    else:
        probs = (1 - uniform_epsilon) * scores_valid + uniform_epsilon
        probs = probs / np.sum(probs)

        sample_indices = np.random.choice(
            len(probs),
            size=min(num_samples, len(probs)),
            replace=False,
            p=probs
        )

        centers = valid_indices[sample_indices]  # (num_samples, 2) [v, u]
        sampled_scores = scores_valid[sample_indices]

        # Convert to (u, v) format
        centers = centers[:, ::-1]  # [u, v]

    return centers, sampled_scores


def visualize_sampling_distribution(
    depth_map,
    centers,
    scores,
    output_path=None
):
    """
    Visualize sampled patch centers overlaid on depth map.

    Args:
        depth_map: (H, W) depth values
        centers: (N, 2) sampled centers [u, v]
        scores: (N,) complexity scores
        output_path: optional path to save visualization

    Returns:
        vis_image: (H, W, 3) BGR image for visualization
    """
    is_torch = isinstance(depth_map, torch.Tensor)

    if is_torch:
        depth_np = depth_map.detach().cpu().numpy()
        centers_np = centers.detach().cpu().numpy() if isinstance(centers, torch.Tensor) else centers
        scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    else:
        depth_np = depth_map
        centers_np = centers
        scores_np = scores

    # Normalize depth for visualization
    depth_vis = depth_np.copy()
    depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
    depth_vis = (depth_vis * 255).astype(np.uint8)

    # Convert to BGR
    vis_image = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

    # Normalize scores for color mapping
    scores_norm = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min() + 1e-8)

    # Draw centers with color indicating score
    for i, (u, v) in enumerate(centers_np):
        u, v = int(u), int(v)
        score = scores_norm[i]

        # Color: green (low score) to red (high score)
        color = (0, int(255 * (1 - score)), int(255 * score))

        cv2.circle(vis_image, (u, v), radius=3, color=color, thickness=-1)
        cv2.circle(vis_image, (u, v), radius=4, color=(255, 255, 255), thickness=1)

    if output_path is not None:
        cv2.imwrite(output_path, vis_image)

    return vis_image


def test_patch_sampling():
    """
    Unit test for patch sampling.
    """
    print("Testing patch sampling...")

    # Create synthetic depth map with structure
    H, W = 256, 256
    depth = np.ones((H, W), dtype=np.float32) * 5.0

    # Add some geometric features
    # Vertical edge
    depth[:, W//2:] = 3.0

    # Horizontal edge
    depth[H//2:, :] = 4.0

    # Corner
    depth[H//4:3*H//4, W//4:3*W//4] = 2.0

    # Add some noise
    depth += np.random.randn(H, W) * 0.1

    # Sample patch centers
    num_samples = 50
    centers, scores = sample_patch_centers(
        depth,
        num_samples=num_samples,
        crop_size=64,
        border_margin=32,
        seed=42
    )

    print(f"  Sampled {len(centers)} centers")
    print(f"  Centers shape: {centers.shape}")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    assert centers.shape == (num_samples, 2), "Centers shape mismatch!"
    assert scores.shape == (num_samples,), "Scores shape mismatch!"

    # Check that centers are within valid range
    assert np.all(centers[:, 0] >= 32) and np.all(centers[:, 0] < W - 32), "Centers out of bounds (u)!"
    assert np.all(centers[:, 1] >= 32) and np.all(centers[:, 1] < H - 32), "Centers out of bounds (v)!"

    print("✓ Patch sampling test passed!\n")


if __name__ == "__main__":
    test_patch_sampling()
