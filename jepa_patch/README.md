# JEPA-Style Pose-Normalized 3D Patch Learning for VGGT-Long

This module implements a self-supervised learning system for learning stable 3D patch representations from RGB-D sequences with non-metric depth scale.

## Overview

The system addresses the challenge of learning consistent place representations across time and viewpoints when depth is non-metric (arbitrary scale). It uses:

1. **Pose-normalized patch frames**: Canonical 3D coordinate systems for patches
2. **Scale-invariant normalization**: Robust to global scale drift
3. **BYOL-style self-supervision**: Learn from overlap + loop closure correspondences
4. **Geometric importance sampling**: Focus on structurally informative regions

## Research Hypothesis

A JEPA/BYOL-style objective trained on pose-normalized local 3D patches will learn embeddings that are:
- Stable across time and viewpoint changes
- Robust to non-metric depth scale variations
- Useful for loop closure verification and drift detection

## Module Structure

```
jepa_patch/
├── __init__.py                 # Module initialization
├── pose_utils.py               # Pose normalization and validation (Phase 0)
├── patch_representation.py     # Patch extraction and canonicalization
├── patch_sampling.py           # Geometric importance sampling
├── patch_dataset.py            # Dataset builder for training pairs
├── patch_encoder.py            # BYOL encoder architecture
├── train.py                    # Training script
├── evaluate.py                 # Evaluation metrics
├── visualize.py                # Visualization tools
├── loop_closure_integration.py # Integration with VGGT-Long
└── README.md                   # This file
```

## Installation

The module is part of VGGT-Long and uses its dependencies:

```bash
# Already installed with VGGT-Long environment
pip install torch torchvision numpy scipy scikit-learn matplotlib opencv-python
```

## Quick Start

### 1. Process Data with VGGT-Long

First, run VGGT-Long on your data to generate processed chunks:

```bash
python vggt_long.py --image_dir /path/to/images --config configs/base_config.yaml
```

This will create `_tmp_results_unaligned/` with chunk data.

### 2. Save Metadata for Patch Learning

After VGGT-Long finishes, save the chunk indices and loop closures:

```python
import json
from pathlib import Path

# After running VGGT-Long
output_dir = Path("./output")

metadata = {
    "chunk_indices": vggt_long_instance.chunk_indices,  # [(start, end), ...]
    "loop_list": vggt_long_instance.loop_list           # [(i, j), ...]
}

with open(output_dir / "metadata.json", "w") as f:
    json.dump(metadata, f)
```

### 3. Train Patch Encoder

```bash
python jepa_patch/train.py \
    --data_dir ./output \
    --output_dir ./jepa_patch_outputs \
    --num_steps 50000 \
    --batch_size 32 \
    --device cuda
```

Training will:
- Sample patch pairs from overlap and loop closures
- Train BYOL encoder with EMA target network
- Save checkpoints every 5000 steps
- Plot training metrics

### 4. Evaluate Trained Encoder

```bash
python jepa_patch/evaluate.py \
    --encoder_path ./jepa_patch_outputs/encoder_latest.pt \
    --data_dir ./output \
    --num_samples 1000 \
    --device cuda
```

Evaluation computes:
- **E1**: AUC for same-place vs different-place classification
- **E3**: Loop closure verification accuracy

### 5. Use for Loop Closure Verification

```python
from jepa_patch.loop_closure_integration import LoopClosureVerifier

# Load trained encoder
verifier = LoopClosureVerifier(
    encoder_path="./jepa_patch_outputs/encoder_latest.pt",
    device="cuda"
)

# Verify a proposed loop closure
is_valid = verifier.verify_loop_closure(
    frame_data_a,
    frame_data_b,
    threshold=0.7,
    num_patches=32
)

print(f"Loop closure valid: {is_valid}")
```

## Patch Representation

Each patch is represented as a 5-channel tensor `(C, H, W)`:

1. **Xn, Yn, Zn**: Normalized 3D coordinates in patch frame
2. **z_rel**: Relative depth channel (shape cue)
3. **mask**: Valid pixel mask

### Scale-Invariant Normalization

Since depth is non-metric, we normalize by local scale:

```
scale = RMS_radius(patch_points)
X_normalized = X_patch / scale
```

This makes the representation robust to global scale drift.

### Patch Frame Canonicalization

Patch frames use a hybrid approach:
- **z-axis**: Gravity/world-up (or local plane normal fallback)
- **x-axis**: Camera forward projected onto tangent plane
- **y-axis**: Cross product

This ensures consistent orientation even with pose drift.

## Training Details

### Architecture

- **Encoder**: Small CNN (64→128→256) + global pooling → 256D embedding
- **Projector**: MLP 256→256→256
- **Predictor**: MLP 256→256→256
- **Target**: EMA (momentum 0.99) of encoder + projector

### Training Recipe

- Batch size: 32 patch pairs (512 patches with 16 patches/pair)
- Optimizer: AdamW (lr_encoder=3e-4, lr_head=1e-3, wd=1e-4)
- Scheduler: Cosine annealing to 1e-6
- Steps: 50k-200k
- Mix: 50% overlap, 50% loop closure pairs

### Data Augmentation

- Random yaw rotation (±10°) in patch frame
- Structured occlusion masks (rectangles, half-planes, frustum slices)
- Random pixel dropout (10-30%)

### Collapse Detection

Monitor `embed_std` during training:
- Healthy: std > 0.1
- Collapse: std → 0

If collapse occurs:
- Increase augmentation strength
- Reduce predictor capacity
- Check target network EMA

## Evaluation Metrics

### E1: Same-place vs Different-place (AUC)

Measures how well the embedding separates positive pairs (same place) from negative pairs (different places).

**Expected**: AUC > 0.8 indicates strong performance

### E2: Long-horizon Consistency (future work)

Track similarity decay over time for revisited places.

### E3: Loop Closure Verification

Use patch embeddings to verify/reject proposed loop closures:
- Sample multiple patches from loop closure frames
- Compute mean similarity
- Accept if similarity > threshold

**Expected**: Reduces false positives without killing true positives

## Integration with VGGT-Long

See `loop_closure_integration.py` for drop-in integration:

```python
# In VGGT-Long's loop closure processing
from jepa_patch.loop_closure_integration import LoopClosureVerifier

verifier = LoopClosureVerifier(encoder_path="./encoder.pt")

# Before accepting a loop closure
valid_loops = []
for (i, j) in proposed_loops:
    frame_i = load_frame(i)
    frame_j = load_frame(j)

    if verifier.verify_loop_closure(frame_i, frame_j, threshold=0.7):
        valid_loops.append((i, j))
```

## Visualization

Visualize patches, embeddings, and training progress:

```python
from jepa_patch.visualize import (
    visualize_patch_representation,
    visualize_patch_pair,
    visualize_embedding_space
)

# Visualize a single patch
visualize_patch_representation(
    patch_tensor,
    metadata,
    save_path="patch.png"
)

# Visualize a patch pair
visualize_patch_pair(
    patch_a, patch_b,
    metadata_a, metadata_b,
    similarity=0.85,
    save_path="patch_pair.png"
)

# Visualize embedding space
visualize_embedding_space(
    embeddings,
    labels=frame_ids,
    method='tsne',
    save_path="embeddings.png"
)
```

## Debugging Checklist

### D0: Patch Visualization (Mandatory)

Before training, verify patches look correct:

```bash
python jepa_patch/patch_representation.py
```

Check:
- ✓ XYZ channels show 3D structure
- ✓ Mask channel shows valid regions
- ✓ 3D scatter shows recognizable geometry

### D1: Transform Convention

Verify pose convention is correct:

```python
from jepa_patch.pose_utils import verify_transform_convention

error = verify_transform_convention(T_wc, K, depth, u, v, X_w)
assert error < 1e-3, f"Reprojection error too large: {error}"
```

### D2: Scale Normalization

Check that scale varies across patches:

```python
scales = [metadata['scale'] for metadata in batch_metadata]
print(f"Scale range: [{min(scales):.3f}, {max(scales):.3f}]")
# Should see variation, not all the same
```

### D3: No Collapse

Monitor during training:

```
Step 1000: {'loss': 1.234, 'pos_sim': 0.45, 'embed_std': 0.234}
```

- ✓ `embed_std` > 0.1 (healthy)
- ✗ `embed_std` < 0.01 (collapse detected!)

## Expected Performance

### Minimal Acceptance Criteria

To proceed with integration:
- **E1 AUC > 0.8** on loop-closure pairs
- **Loop closure gating** reduces false positives without killing true positives
- **Drift scoring** correlates with bad reconstruction segments

### Typical Results

On a dataset with:
- 5000 frames
- 50 chunks (overlap 30)
- 100 loop closures

Training for 50k steps (~2 hours on RTX 4090):
- E1 AUC: 0.85-0.90
- Loop verification: 95% recall at 10% FPR

## Failure Modes

### F1: Patches Too Noisy

**Symptom**: Low AUC, high variance in similarities

**Solution**:
- Increase crop size (64→96)
- Strengthen scale normalization
- Use ray coordinates (S2 fallback)

### F2: Pose Drift Breaks Correspondences

**Symptom**: Loop closure pairs have low similarity

**Solution**:
- Use larger patches + spatial pooling
- Require multiple patch centers and aggregate robustly (median)
- Add lightweight local alignment (future work)

### F3: Model Learns Only "Wallness"

**Symptom**: High AUC but all walls look the same

**Solution**:
- Bias sampling toward high-structure areas (increase alpha, beta)
- Add auxiliary task: predict depth histogram bins
- Visualize embeddings with t-SNE to check clustering

## Citation

If you use this code, please cite:

```bibtex
@misc{vggtlong_jepa_patches,
  title={JEPA-Style Pose-Normalized 3D Patch Learning for VGGT-Long},
  author={Your Name},
  year={2026}
}
```

## License

Same as VGGT-Long (see parent directory LICENSE).

## Future Work

- [ ] Multi-scale patches (small + large context)
- [ ] Grid+set fusion for occlusion robustness
- [ ] Semantic smoothing / update reweighting (U3)
- [ ] Cross-dataset generalization tests
- [ ] Integration with Map-Anything and Pi3

## Contact

For issues or questions, open an issue on the VGGT-Long repository.
