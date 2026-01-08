"""
Visualization tools for patch learning.

Provides functions to visualize:
- Patch representations
- Sampling distributions
- Patch correspondences
- Embedding similarities
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import torch


def visualize_patch_representation(
    patch_tensor,
    metadata,
    save_path=None,
    show=False
):
    """
    Visualize a patch representation.

    Args:
        patch_tensor: (C, H, W) patch tensor
        metadata: dict with patch metadata
        save_path: optional path to save figure
        show: whether to display the figure
    """
    is_torch = isinstance(patch_tensor, torch.Tensor)

    if is_torch:
        patch_np = patch_tensor.detach().cpu().numpy()
    else:
        patch_np = patch_tensor

    C, H, W = patch_np.shape

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: XYZ channels
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(patch_np[0], cmap='RdBu', vmin=-3, vmax=3)
    ax1.set_title('X (normalized)')
    ax1.axis('off')
    plt.colorbar(ax1.images[0], ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(patch_np[1], cmap='RdBu', vmin=-3, vmax=3)
    ax2.set_title('Y (normalized)')
    ax2.axis('off')
    plt.colorbar(ax2.images[0], ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(patch_np[2], cmap='RdBu', vmin=-3, vmax=3)
    ax3.set_title('Z (normalized)')
    ax3.axis('off')
    plt.colorbar(ax3.images[0], ax=ax3, fraction=0.046, pad=0.04)

    # Plot 2: Relative depth channel (if present)
    if C >= 4:
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.imshow(patch_np[3], cmap='viridis')
        ax4.set_title('Z_rel (relative depth)')
        ax4.axis('off')
        plt.colorbar(ax4.images[0], ax=ax4, fraction=0.046, pad=0.04)

    # Plot 3: Mask channel
    mask_idx = 4 if C >= 5 else C - 1
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(patch_np[mask_idx], cmap='gray')
    ax5.set_title('Valid mask')
    ax5.axis('off')

    # Plot 4: 3D scatter of points
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')

    # Extract XYZ and mask
    X = patch_np[0]
    Y = patch_np[1]
    Z = patch_np[2]
    mask = patch_np[mask_idx] > 0.5

    # Downsample for visualization
    stride = max(1, H // 32)
    X_ds = X[::stride, ::stride][mask[::stride, ::stride]]
    Y_ds = Y[::stride, ::stride][mask[::stride, ::stride]]
    Z_ds = Z[::stride, ::stride][mask[::stride, ::stride]]

    ax6.scatter(X_ds, Y_ds, Z_ds, c=Z_ds, cmap='viridis', s=1)
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.set_title('3D points in patch frame')

    # Metadata text
    info_text = f"Scale: {metadata.get('scale', 'N/A'):.3f}\n"
    info_text += f"Valid pixels: {metadata.get('valid_pixel_ratio', 'N/A'):.2%}\n"
    info_text += f"Frame confidence: {metadata.get('frame_confidence', 'N/A'):.3f}"

    plt.figtext(0.02, 0.02, info_text, fontsize=10, family='monospace')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def visualize_patch_pair(
    patch_a,
    patch_b,
    metadata_a,
    metadata_b,
    similarity=None,
    save_path=None,
    show=False
):
    """
    Visualize a pair of corresponding patches.

    Args:
        patch_a, patch_b: (C, H, W) patch tensors
        metadata_a, metadata_b: metadata dicts
        similarity: optional similarity score between patches
        save_path: optional path to save figure
        show: whether to display the figure
    """
    is_torch = isinstance(patch_a, torch.Tensor)

    if is_torch:
        patch_a = patch_a.detach().cpu().numpy()
        patch_b = patch_b.detach().cpu().numpy()

    C, H, W = patch_a.shape

    fig = plt.figure(figsize=(18, 10))

    # Patch A
    for i, (title, idx) in enumerate([('X', 0), ('Y', 1), ('Z', 2), ('Mask', -1)]):
        ax = fig.add_subplot(3, 4, i + 1)
        ax.imshow(patch_a[idx], cmap='RdBu' if i < 3 else 'gray')
        ax.set_title(f'Patch A: {title}')
        ax.axis('off')

    # Patch B
    for i, (title, idx) in enumerate([('X', 0), ('Y', 1), ('Z', 2), ('Mask', -1)]):
        ax = fig.add_subplot(3, 4, i + 5)
        ax.imshow(patch_b[idx], cmap='RdBu' if i < 3 else 'gray')
        ax.set_title(f'Patch B: {title}')
        ax.axis('off')

    # 3D scatter for both
    ax_3d_a = fig.add_subplot(3, 4, 9, projection='3d')
    mask_a = patch_a[-1] > 0.5
    stride = max(1, H // 32)
    X_a = patch_a[0][::stride, ::stride][mask_a[::stride, ::stride]]
    Y_a = patch_a[1][::stride, ::stride][mask_a[::stride, ::stride]]
    Z_a = patch_a[2][::stride, ::stride][mask_a[::stride, ::stride]]
    ax_3d_a.scatter(X_a, Y_a, Z_a, c=Z_a, cmap='viridis', s=1)
    ax_3d_a.set_title('Patch A (3D)')

    ax_3d_b = fig.add_subplot(3, 4, 10, projection='3d')
    mask_b = patch_b[-1] > 0.5
    X_b = patch_b[0][::stride, ::stride][mask_b[::stride, ::stride]]
    Y_b = patch_b[1][::stride, ::stride][mask_b[::stride, ::stride]]
    Z_b = patch_b[2][::stride, ::stride][mask_b[::stride, ::stride]]
    ax_3d_b.scatter(X_b, Y_b, Z_b, c=Z_b, cmap='viridis', s=1)
    ax_3d_b.set_title('Patch B (3D)')

    # Metadata and similarity
    info_text = f"Patch A - Scale: {metadata_a.get('scale', 'N/A'):.3f}, "
    info_text += f"Valid: {metadata_a.get('valid_pixel_ratio', 'N/A'):.2%}\n"
    info_text += f"Patch B - Scale: {metadata_b.get('scale', 'N/A'):.3f}, "
    info_text += f"Valid: {metadata_b.get('valid_pixel_ratio', 'N/A'):.2%}\n"

    if similarity is not None:
        info_text += f"\nSimilarity: {similarity:.4f}"

    plt.figtext(0.02, 0.02, info_text, fontsize=10, family='monospace')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def visualize_embedding_space(
    embeddings,
    labels=None,
    method='tsne',
    save_path=None,
    show=False
):
    """
    Visualize embedding space using dimensionality reduction.

    Args:
        embeddings: (N, D) embeddings
        labels: (N,) optional labels for coloring
        method: 'tsne' or 'pca'
        save_path: optional path to save figure
        show: whether to display the figure
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    is_torch = isinstance(embeddings, torch.Tensor)

    if is_torch:
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    embeddings_2d = reducer.fit_transform(embeddings_np)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=20
        )

    ax.set_title(f'Embedding Space ({method.upper()})')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def visualize_similarity_matrix(
    embeddings,
    labels=None,
    save_path=None,
    show=False
):
    """
    Visualize pairwise similarity matrix.

    Args:
        embeddings: (N, D) embeddings
        labels: (N,) optional labels
        save_path: optional path to save figure
        show: whether to display the figure
    """
    is_torch = isinstance(embeddings, torch.Tensor)

    if is_torch:
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    # Normalize embeddings
    embeddings_norm = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)

    # Compute similarity matrix
    sim_matrix = embeddings_norm @ embeddings_norm.T

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_title('Pairwise Similarity Matrix')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')

    if labels is not None:
        # Add lines to separate different labels
        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) > 0:
                ax.axhline(indices[0] - 0.5, color='blue', linewidth=0.5, alpha=0.5)
                ax.axvline(indices[0] - 0.5, color='blue', linewidth=0.5, alpha=0.5)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    print("Visualization tools loaded.")
    print("Use these functions to visualize patches and embeddings during training/evaluation.")
