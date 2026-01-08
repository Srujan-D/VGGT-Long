"""
Loop closure verification integration for VGGT-Long.

This module provides a drop-in component for verifying loop closures
using learned patch embeddings.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from .patch_encoder import BYOL_PatchEncoder
from .patch_representation import extract_patch_representation
from .patch_sampling import sample_patch_centers


class LoopClosureVerifier:
    """
    Loop closure verifier using trained patch encoder.

    Can be plugged into VGGT-Long's loop closure pipeline to filter
    false positives before running SIM(3) optimization.
    """

    def __init__(
        self,
        encoder_path,
        device='cuda',
        threshold=0.7,
        num_patches=32,
        min_valid_patches=16,
        aggregation='mean'
    ):
        """
        Args:
            encoder_path: path to trained encoder checkpoint
            device: device to use
            threshold: similarity threshold for verification (default: 0.7)
            num_patches: number of patches to sample per frame (default: 32)
            min_valid_patches: minimum number of valid patch pairs required (default: 16)
            aggregation: 'mean', 'median', or 'min' for aggregating patch similarities
        """
        self.device = torch.device(device)
        self.threshold = threshold
        self.num_patches = num_patches
        self.min_valid_patches = min_valid_patches
        self.aggregation = aggregation

        # Load encoder
        checkpoint = torch.load(encoder_path, map_location=self.device)

        config = checkpoint.get('config', {})

        self.encoder = BYOL_PatchEncoder(
            in_channels=config.get('model', {}).get('in_channels', 5),
            encoder_hidden_dims=config.get('model', {}).get('encoder_hidden_dims', [64, 128, 256]),
            embed_dim=config.get('model', {}).get('embed_dim', 256),
            proj_dim=config.get('model', {}).get('proj_dim', 256),
            pred_dim=config.get('model', {}).get('pred_dim', 256)
        ).to(self.device)

        if 'encoder_state_dict' in checkpoint:
            self.encoder.online_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint['model_state_dict'])

        self.encoder.eval()

        print(f"Loaded patch encoder from {encoder_path}")
        print(f"Verification threshold: {threshold}")

    @torch.no_grad()
    def encode_patches(self, patches):
        """
        Encode a batch of patches.

        Args:
            patches: (B, C, H, W) patch tensors

        Returns:
            embeddings: (B, D) embeddings
        """
        if not isinstance(patches, torch.Tensor):
            patches = torch.from_numpy(patches).float()

        patches = patches.to(self.device)
        embeddings = self.encoder.online_encoder(patches)
        return embeddings

    @torch.no_grad()
    def compute_similarity(self, emb_a, emb_b):
        """
        Compute cosine similarity between embeddings.

        Args:
            emb_a: (N, D) embeddings
            emb_b: (M, D) embeddings

        Returns:
            similarity: (N, M) similarity matrix
        """
        emb_a = F.normalize(emb_a, dim=-1, p=2)
        emb_b = F.normalize(emb_b, dim=-1, p=2)

        similarity = torch.mm(emb_a, emb_b.T)

        return similarity

    def extract_patches_from_frame(
        self,
        depth,
        intrinsic,
        extrinsic,
        num_patches=None,
        patch_size=64
    ):
        """
        Extract patches from a single frame.

        Args:
            depth: (H, W) depth map
            intrinsic: (3, 3) camera intrinsics
            extrinsic: (4, 4) camera extrinsic (C2W)
            num_patches: number of patches to sample (default: self.num_patches)
            patch_size: patch size (default: 64)

        Returns:
            patches: (N, C, H, W) patch tensors
            metadata: list of metadata dicts
        """
        if num_patches is None:
            num_patches = self.num_patches

        # Sample patch centers
        try:
            centers, scores = sample_patch_centers(
                depth,
                num_samples=num_patches,
                crop_size=patch_size,
                border_margin=32,
                seed=None
            )
        except ValueError as e:
            # No valid pixels
            return None, None

        patches = []
        metadata_list = []

        for i in range(len(centers)):
            u, v = centers[i]

            # Get depth at center
            depth_center = depth[int(v), int(u)]
            if depth_center <= 0:
                continue

            # Backproject center to world
            K_inv = np.linalg.inv(intrinsic)
            pixel_homo = np.array([u, v, 1.0])
            X_c = depth_center * (K_inv @ pixel_homo)

            R_wc = extrinsic[:3, :3]
            t_wc = extrinsic[:3, 3]
            X_w = R_wc @ X_c + t_wc

            # Extract crop
            half = patch_size // 2
            u_start = max(0, int(u) - half)
            u_end = min(depth.shape[1], int(u) + half)
            v_start = max(0, int(v) - half)
            v_end = min(depth.shape[0], int(v) + half)

            depth_crop = depth[v_start:v_end, u_start:u_end]

            # Pad if needed
            if depth_crop.shape[0] < patch_size or depth_crop.shape[1] < patch_size:
                pad_h = max(0, patch_size - depth_crop.shape[0])
                pad_w = max(0, patch_size - depth_crop.shape[1])
                depth_crop = np.pad(depth_crop, ((0, pad_h), (0, pad_w)), mode='constant')

            depth_crop = depth_crop[:patch_size, :patch_size]

            # Dummy RGB (not used)
            rgb_crop = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

            try:
                # Extract patch representation
                patch, meta = extract_patch_representation(
                    rgb_crop,
                    depth_crop,
                    intrinsic,
                    extrinsic,
                    X_w,
                    include_rgb=False,
                    include_relative_depth=True
                )

                patches.append(patch)
                metadata_list.append(meta)

            except Exception as e:
                # Skip if patch extraction fails
                continue

        if len(patches) == 0:
            return None, None

        patches = np.stack(patches, axis=0)  # (N, C, H, W)

        return patches, metadata_list

    def verify_loop_closure(
        self,
        frame_a_data,
        frame_b_data,
        threshold=None,
        num_patches=None
    ):
        """
        Verify a proposed loop closure between two frames.

        Args:
            frame_a_data: dict with keys 'depth', 'intrinsic', 'extrinsic'
            frame_b_data: dict with keys 'depth', 'intrinsic', 'extrinsic'
            threshold: similarity threshold (default: self.threshold)
            num_patches: number of patches to sample (default: self.num_patches)

        Returns:
            is_valid: bool indicating whether loop closure is valid
            score: float similarity score
            details: dict with additional information
        """
        if threshold is None:
            threshold = self.threshold

        if num_patches is None:
            num_patches = self.num_patches

        # Extract patches from both frames
        patches_a, meta_a = self.extract_patches_from_frame(
            frame_a_data['depth'],
            frame_a_data['intrinsic'],
            frame_a_data['extrinsic'],
            num_patches=num_patches
        )

        patches_b, meta_b = self.extract_patches_from_frame(
            frame_b_data['depth'],
            frame_b_data['intrinsic'],
            frame_b_data['extrinsic'],
            num_patches=num_patches
        )

        if patches_a is None or patches_b is None:
            # Failed to extract patches
            return False, 0.0, {'error': 'Failed to extract patches'}

        # Encode patches
        emb_a = self.encode_patches(patches_a)
        emb_b = self.encode_patches(patches_b)

        # Compute pairwise similarities
        similarities = self.compute_similarity(emb_a, emb_b)  # (N, M)

        # Get best match for each patch in A
        best_matches, _ = torch.max(similarities, dim=1)  # (N,)

        # Aggregate similarity scores
        if self.aggregation == 'mean':
            score = torch.mean(best_matches).item()
        elif self.aggregation == 'median':
            score = torch.median(best_matches).item()
        elif self.aggregation == 'min':
            score = torch.min(best_matches).item()
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        is_valid = score >= threshold

        details = {
            'num_patches_a': len(patches_a),
            'num_patches_b': len(patches_b),
            'score': score,
            'threshold': threshold,
            'similarities_mean': torch.mean(best_matches).item(),
            'similarities_std': torch.std(best_matches).item(),
            'similarities_min': torch.min(best_matches).item(),
            'similarities_max': torch.max(best_matches).item()
        }

        return is_valid, score, details

    def filter_loop_closures(
        self,
        loop_list,
        frame_data_loader,
        threshold=None,
        verbose=True
    ):
        """
        Filter a list of proposed loop closures.

        Args:
            loop_list: list of (frame_i, frame_j) tuples
            frame_data_loader: function that takes frame index and returns frame data dict
            threshold: similarity threshold (default: self.threshold)
            verbose: whether to print progress

        Returns:
            valid_loops: list of (frame_i, frame_j) tuples that passed verification
            scores: list of similarity scores for valid loops
            rejected_loops: list of (frame_i, frame_j) tuples that were rejected
        """
        if threshold is None:
            threshold = self.threshold

        valid_loops = []
        scores = []
        rejected_loops = []

        for i, (frame_i, frame_j) in enumerate(loop_list):
            if verbose and i % 10 == 0:
                print(f"Verifying loop closure {i+1}/{len(loop_list)}: ({frame_i}, {frame_j})")

            try:
                frame_a_data = frame_data_loader(frame_i)
                frame_b_data = frame_data_loader(frame_j)

                is_valid, score, details = self.verify_loop_closure(
                    frame_a_data,
                    frame_b_data,
                    threshold=threshold
                )

                if is_valid:
                    valid_loops.append((frame_i, frame_j))
                    scores.append(score)
                    if verbose:
                        print(f"  ✓ Valid (score={score:.4f})")
                else:
                    rejected_loops.append((frame_i, frame_j))
                    if verbose:
                        print(f"  ✗ Rejected (score={score:.4f})")

            except Exception as e:
                if verbose:
                    print(f"  ⚠ Error: {e}")
                rejected_loops.append((frame_i, frame_j))

        if verbose:
            print(f"\nFiltering complete:")
            print(f"  Valid: {len(valid_loops)}/{len(loop_list)} ({100*len(valid_loops)/len(loop_list):.1f}%)")
            print(f"  Rejected: {len(rejected_loops)}/{len(loop_list)} ({100*len(rejected_loops)/len(loop_list):.1f}%)")

        return valid_loops, scores, rejected_loops


class DriftScorer:
    """
    Drift/failure detection using patch embeddings.

    Monitors embedding similarity in overlap regions to detect
    when chunk alignment may be failing.
    """

    def __init__(
        self,
        verifier,
        low_similarity_threshold=0.5,
        high_variance_threshold=0.3
    ):
        """
        Args:
            verifier: LoopClosureVerifier instance
            low_similarity_threshold: threshold for low similarity warning
            high_variance_threshold: threshold for high variance warning
        """
        self.verifier = verifier
        self.low_similarity_threshold = low_similarity_threshold
        self.high_variance_threshold = high_variance_threshold

    def score_overlap_region(
        self,
        frame_data_list,
        num_patches=32
    ):
        """
        Score an overlap region between consecutive chunks.

        Args:
            frame_data_list: list of frame data dicts in overlap region
            num_patches: number of patches to sample per frame

        Returns:
            score: float quality score (higher is better)
            details: dict with diagnostics
        """
        if len(frame_data_list) < 2:
            return 1.0, {'error': 'Not enough frames'}

        # Extract patches from all frames
        all_embeddings = []

        for frame_data in frame_data_list:
            patches, _ = self.verifier.extract_patches_from_frame(
                frame_data['depth'],
                frame_data['intrinsic'],
                frame_data['extrinsic'],
                num_patches=num_patches
            )

            if patches is not None:
                emb = self.verifier.encode_patches(patches)
                all_embeddings.append(emb)

        if len(all_embeddings) < 2:
            return 0.0, {'error': 'Failed to extract patches'}

        # Compute pairwise similarities
        similarities = []

        for i in range(len(all_embeddings) - 1):
            sim_matrix = self.verifier.compute_similarity(
                all_embeddings[i],
                all_embeddings[i + 1]
            )
            mean_sim = torch.mean(sim_matrix).item()
            similarities.append(mean_sim)

        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)

        # Score: penalize low similarity or high variance
        score = mean_similarity

        if mean_similarity < self.low_similarity_threshold:
            score *= 0.5  # Penalize
        if std_similarity > self.high_variance_threshold:
            score *= 0.8  # Penalize

        details = {
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'num_frames': len(frame_data_list),
            'num_embeddings': len(all_embeddings),
            'warning': mean_similarity < self.low_similarity_threshold or
                      std_similarity > self.high_variance_threshold
        }

        return score, details


def integrate_with_vggt_long(vggt_long_instance, encoder_path, threshold=0.7):
    """
    Integrate patch-based loop closure verification with VGGT-Long.

    This function wraps VGGT-Long's loop closure detection to add
    patch-based verification.

    Args:
        vggt_long_instance: instance of VGGT_Long class
        encoder_path: path to trained encoder
        threshold: verification threshold

    Returns:
        modified vggt_long_instance with integrated verifier
    """
    # Create verifier
    verifier = LoopClosureVerifier(
        encoder_path=encoder_path,
        threshold=threshold
    )

    # Store original loop list
    original_loop_list = vggt_long_instance.loop_list.copy()

    # Frame data loader
    def frame_data_loader(frame_idx):
        # Find which chunk this frame belongs to
        chunk_idx = None
        for ci, (start, end) in enumerate(vggt_long_instance.chunk_indices):
            if start <= frame_idx < end:
                chunk_idx = ci
                break

        if chunk_idx is None:
            raise ValueError(f"Frame {frame_idx} not found in any chunk")

        # Load chunk data
        chunk_file = Path(vggt_long_instance.result_unaligned_dir) / f"chunk_{chunk_idx}.npy"
        chunk_data = np.load(chunk_file, allow_pickle=True).item()

        rel_idx = frame_idx - vggt_long_instance.chunk_indices[chunk_idx][0]

        return {
            'depth': chunk_data['depth'][rel_idx],
            'intrinsic': chunk_data['intrinsic'][rel_idx],
            'extrinsic': chunk_data['extrinsic'][rel_idx]
        }

    # Filter loop closures
    print("\n" + "="*60)
    print("Filtering loop closures with patch embeddings...")
    print("="*60)

    valid_loops, scores, rejected_loops = verifier.filter_loop_closures(
        original_loop_list,
        frame_data_loader,
        threshold=threshold,
        verbose=True
    )

    # Update VGGT-Long's loop list
    vggt_long_instance.loop_list = valid_loops

    print(f"\nFiltered {len(rejected_loops)} false positive loop closures")
    print(f"Retained {len(valid_loops)} valid loop closures")

    return vggt_long_instance, verifier


if __name__ == "__main__":
    print("Loop closure integration module loaded.")
    print("Use LoopClosureVerifier to verify loop closures or DriftScorer to detect drift.")
