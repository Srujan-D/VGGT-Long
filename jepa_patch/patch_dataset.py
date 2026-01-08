"""
Patch pair dataset builder for JEPA/BYOL training.

Generates training samples from VGGT-Long chunk data:
- Overlap pairs (short-horizon positives)
- Loop-closure pairs (long-horizon positives)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
import pickle

from .patch_representation import extract_patch_representation
from .patch_sampling import sample_patch_centers


class PatchPairDataset(Dataset):
    """
    Dataset that generates patch pairs from VGGT-Long processed data.

    Samples positive pairs from:
    1. Overlap regions between adjacent chunks
    2. Loop closure correspondences
    """

    def __init__(
        self,
        data_dir,
        chunk_indices,
        loop_list=None,
        num_patches_per_pair=16,
        patch_size=64,
        overlap_weight=0.5,
        loop_weight=0.5,
        augmentation=True,
        seed=42
    ):
        """
        Args:
            data_dir: path to VGGT-Long output directory with processed chunks
            chunk_indices: list of (start_idx, end_idx) for each chunk
            loop_list: list of (frame_i, frame_j) loop closure pairs
            num_patches_per_pair: number of patches to sample per frame pair
            patch_size: size of patch crops
            overlap_weight: weight for overlap pairs in sampling
            loop_weight: weight for loop closure pairs in sampling
            augmentation: whether to apply data augmentation
            seed: random seed
        """
        self.data_dir = Path(data_dir)
        self.chunk_indices = chunk_indices
        self.loop_list = loop_list if loop_list is not None else []
        self.num_patches = num_patches_per_pair
        self.patch_size = patch_size
        self.overlap_weight = overlap_weight
        self.loop_weight = loop_weight
        self.augmentation = augmentation
        self.seed = seed

        self.rng = np.random.RandomState(seed)

        # Build list of valid pairs
        self._build_pair_list()

    def _build_pair_list(self):
        """
        Build a list of all valid frame pairs (overlap + loop closure).
        """
        self.pairs = []

        # 1. Overlap pairs from adjacent chunks
        for chunk_idx in range(len(self.chunk_indices) - 1):
            chunk_a_start, chunk_a_end = self.chunk_indices[chunk_idx]
            chunk_b_start, chunk_b_end = self.chunk_indices[chunk_idx + 1]

            # Overlap region
            overlap_start = chunk_b_start
            overlap_end = min(chunk_a_end, chunk_b_end)

            if overlap_end > overlap_start:
                # Sample pairs within overlap
                for t in range(overlap_start, overlap_end - 1):
                    # Pair with next frame (small delta)
                    t_prime = t + self.rng.randint(1, min(5, overlap_end - t))
                    if t_prime < overlap_end:
                        self.pairs.append({
                            'type': 'overlap',
                            'frame_a': t,
                            'frame_b': t_prime,
                            'chunk_a': chunk_idx,
                            'chunk_b': chunk_idx + 1
                        })

        # 2. Loop closure pairs
        for (frame_i, frame_j) in self.loop_list:
            # Find which chunks these frames belong to
            chunk_i = self._find_chunk_for_frame(frame_i)
            chunk_j = self._find_chunk_for_frame(frame_j)

            if chunk_i is not None and chunk_j is not None:
                self.pairs.append({
                    'type': 'loop',
                    'frame_a': frame_i,
                    'frame_b': frame_j,
                    'chunk_a': chunk_i,
                    'chunk_b': chunk_j
                })

        print(f"Built {len(self.pairs)} patch pairs:")
        overlap_count = sum(1 for p in self.pairs if p['type'] == 'overlap')
        loop_count = sum(1 for p in self.pairs if p['type'] == 'loop')
        print(f"  - {overlap_count} overlap pairs")
        print(f"  - {loop_count} loop closure pairs")

    def _find_chunk_for_frame(self, frame_idx):
        """Find which chunk a frame belongs to."""
        for chunk_idx, (start, end) in enumerate(self.chunk_indices):
            if start <= frame_idx < end:
                return chunk_idx
        return None

    def __len__(self):
        return len(self.pairs)

    def _load_frame_data(self, chunk_idx, frame_idx):
        """
        Load frame data from processed chunk file.

        Returns:
            dict with keys: 'rgb', 'depth', 'depth_conf', 'extrinsic', 'intrinsic', 'world_points'
        """
        chunk_file = self.data_dir / f"_tmp_results_unaligned/chunk_{chunk_idx}.npy"

        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

        chunk_data = np.load(chunk_file, allow_pickle=True).item()

        # Get relative frame index within chunk
        chunk_start, chunk_end = self.chunk_indices[chunk_idx]
        rel_idx = frame_idx - chunk_start

        return {
            'depth': chunk_data['depth'][rel_idx],
            'depth_conf': chunk_data['depth_conf'][rel_idx],
            'extrinsic': chunk_data['extrinsic'][rel_idx],  # C2W
            'intrinsic': chunk_data['intrinsic'][rel_idx],
            'world_points': chunk_data['world_points'][rel_idx],
            'images': chunk_data['images'][rel_idx]  # Might be downsampled
        }

    def _crop_patch(self, image, center_uv, crop_size):
        """
        Extract a crop from an image centered at (u, v).

        Args:
            image: (H, W, C) or (H, W) image
            center_uv: (u, v) center coordinates
            crop_size: size of crop

        Returns:
            crop: (crop_size, crop_size, C) or (crop_size, crop_size)
        """
        u, v = int(center_uv[0]), int(center_uv[1])
        half = crop_size // 2

        H, W = image.shape[:2]

        # Ensure crop is within bounds
        u_start = max(0, u - half)
        u_end = min(W, u + half)
        v_start = max(0, v - half)
        v_end = min(H, v + half)

        crop = image[v_start:v_end, u_start:u_end]

        # Pad if needed
        if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
            if image.ndim == 3:
                pad_h = max(0, crop_size - crop.shape[0])
                pad_w = max(0, crop_size - crop.shape[1])
                crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            else:
                pad_h = max(0, crop_size - crop.shape[0])
                pad_w = max(0, crop_size - crop.shape[1])
                crop = np.pad(crop, ((0, pad_h), (0, pad_w)), mode='constant')

        return crop[:crop_size, :crop_size]

    def _apply_augmentation(self, patch_tensor, metadata):
        """
        Apply data augmentation to a patch.

        Augmentations:
        - Random yaw rotation in patch frame (±10°)
        - Structured occlusion masks
        - Random pixel dropout
        """
        if not self.augmentation:
            return patch_tensor, metadata

        C, H, W = patch_tensor.shape

        # 1. Random yaw rotation (rotate XY coordinates in patch frame)
        if self.rng.rand() < 0.5:
            angle_deg = self.rng.uniform(-10, 10)
            angle_rad = np.deg2rad(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Rotate X, Y channels (channels 0, 1)
            X = patch_tensor[0].copy()
            Y = patch_tensor[1].copy()
            patch_tensor[0] = cos_a * X - sin_a * Y
            patch_tensor[1] = sin_a * X + cos_a * Y

        # 2. Structured occlusion masks
        if self.rng.rand() < 0.3:
            mask_type = self.rng.choice(['rectangle', 'half_plane', 'frustum'])

            if mask_type == 'rectangle':
                # Random rectangle
                w = self.rng.randint(W // 4, W // 2)
                h = self.rng.randint(H // 4, H // 2)
                x = self.rng.randint(0, W - w)
                y = self.rng.randint(0, H - h)
                patch_tensor[:, y:y+h, x:x+w] = 0

            elif mask_type == 'half_plane':
                # Mask half of the patch
                if self.rng.rand() < 0.5:
                    patch_tensor[:, :, :W//2] = 0  # Left half
                else:
                    patch_tensor[:, H//2:, :] = 0  # Bottom half

            elif mask_type == 'frustum':
                # Mask columns or rows
                if self.rng.rand() < 0.5:
                    col_start = self.rng.randint(0, W - W//4)
                    col_end = col_start + W // 4
                    patch_tensor[:, :, col_start:col_end] = 0
                else:
                    row_start = self.rng.randint(0, H - H//4)
                    row_end = row_start + H // 4
                    patch_tensor[:, row_start:row_end, :] = 0

        # 3. Random pixel dropout (10-30%)
        if self.rng.rand() < 0.5:
            dropout_rate = self.rng.uniform(0.1, 0.3)
            dropout_mask = self.rng.rand(H, W) > dropout_rate
            # Apply to all channels except mask channel (last channel)
            for c in range(C - 1):
                patch_tensor[c] = patch_tensor[c] * dropout_mask

        return patch_tensor, metadata

    def __getitem__(self, idx):
        """
        Get a patch pair sample.

        Returns:
            dict with keys:
                - patch_a: (num_patches, C, H, W) patches from frame A
                - patch_b: (num_patches, C, H, W) patches from frame B
                - metadata_a: list of metadata dicts for patches A
                - metadata_b: list of metadata dicts for patches B
                - pair_type: 'overlap' or 'loop'
        """
        pair_info = self.pairs[idx]

        # Load frame data
        frame_a_data = self._load_frame_data(pair_info['chunk_a'], pair_info['frame_a'])
        frame_b_data = self._load_frame_data(pair_info['chunk_b'], pair_info['frame_b'])

        # Sample patch centers in frame A
        centers_a, scores_a = sample_patch_centers(
            frame_a_data['depth'],
            num_samples=self.num_patches,
            crop_size=self.patch_size,
            seed=self.rng.randint(0, 2**31)
        )

        # For each center in A, find corresponding center in B by projecting 3D point
        patches_a = []
        patches_b = []
        metadata_a_list = []
        metadata_b_list = []

        for i in range(len(centers_a)):
            u_a, v_a = centers_a[i]

            # Backproject center to 3D world space
            depth_a = frame_a_data['depth'][int(v_a), int(u_a)]
            if depth_a <= 0:
                continue  # Invalid depth

            K_a = frame_a_data['intrinsic']
            T_wc_a = frame_a_data['extrinsic']

            # Backproject to camera frame
            K_a_inv = np.linalg.inv(K_a)
            pixel_homo = np.array([u_a, v_a, 1.0])
            X_c_a = depth_a * (K_a_inv @ pixel_homo)

            # Transform to world frame
            R_wc_a = T_wc_a[:3, :3]
            t_wc_a = T_wc_a[:3, 3]
            X_w = R_wc_a @ X_c_a + t_wc_a

            # Project to frame B
            T_wc_b = frame_b_data['extrinsic']
            T_cw_b = np.linalg.inv(T_wc_b)
            R_cw_b = T_cw_b[:3, :3]
            t_cw_b = T_cw_b[:3, 3]

            X_c_b = R_cw_b @ X_w + t_cw_b

            if X_c_b[2] <= 0:
                continue  # Behind camera

            K_b = frame_b_data['intrinsic']
            uv_b_homo = K_b @ X_c_b
            u_b = uv_b_homo[0] / uv_b_homo[2]
            v_b = uv_b_homo[1] / uv_b_homo[2]

            H_b, W_b = frame_b_data['depth'].shape
            if not (0 <= u_b < W_b and 0 <= v_b < H_b):
                continue  # Out of bounds

            # Extract crops
            depth_crop_a = self._crop_patch(frame_a_data['depth'], (u_a, v_a), self.patch_size)
            depth_crop_b = self._crop_patch(frame_b_data['depth'], (u_b, v_b), self.patch_size)

            # Get RGB (if available - for now use placeholder)
            rgb_crop_a = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            rgb_crop_b = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)

            try:
                # Extract patch representation
                patch_a, meta_a = extract_patch_representation(
                    rgb_crop_a,
                    depth_crop_a,
                    K_a,
                    T_wc_a,
                    X_w,
                    include_rgb=False,
                    include_relative_depth=True
                )

                patch_b, meta_b = extract_patch_representation(
                    rgb_crop_b,
                    depth_crop_b,
                    K_b,
                    T_wc_b,
                    X_w,
                    include_rgb=False,
                    include_relative_depth=True
                )

                # Apply augmentation
                patch_a, meta_a = self._apply_augmentation(patch_a, meta_a)
                patch_b, meta_b = self._apply_augmentation(patch_b, meta_b)

                patches_a.append(patch_a)
                patches_b.append(patch_b)
                metadata_a_list.append(meta_a)
                metadata_b_list.append(meta_b)

            except Exception as e:
                # Skip if patch extraction fails
                continue

        if len(patches_a) == 0:
            # Fallback: return dummy data (will be filtered during training)
            dummy_patch = np.zeros((5, self.patch_size, self.patch_size), dtype=np.float32)
            return {
                'patch_a': torch.from_numpy(np.stack([dummy_patch])),
                'patch_b': torch.from_numpy(np.stack([dummy_patch])),
                'metadata_a': [{}],
                'metadata_b': [{}],
                'pair_type': pair_info['type'],
                'valid': False
            }

        # Stack patches
        patches_a = np.stack(patches_a, axis=0)  # (N, C, H, W)
        patches_b = np.stack(patches_b, axis=0)

        return {
            'patch_a': torch.from_numpy(patches_a).float(),
            'patch_b': torch.from_numpy(patches_b).float(),
            'metadata_a': metadata_a_list,
            'metadata_b': metadata_b_list,
            'pair_type': pair_info['type'],
            'valid': True
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable number of patches per sample.
    """
    # Filter out invalid samples
    valid_batch = [item for item in batch if item.get('valid', True)]

    if len(valid_batch) == 0:
        # Return dummy batch
        dummy = batch[0]
        return dummy

    # Concatenate all patches from all samples in the batch
    all_patches_a = torch.cat([item['patch_a'] for item in valid_batch], dim=0)
    all_patches_b = torch.cat([item['patch_b'] for item in valid_batch], dim=0)

    return {
        'patch_a': all_patches_a,
        'patch_b': all_patches_b,
        'pair_types': [item['pair_type'] for item in valid_batch],
        'valid': True
    }


if __name__ == "__main__":
    # Example usage
    print("Patch dataset module loaded.")
    print("To use: create PatchPairDataset with your VGGT-Long data directory.")
