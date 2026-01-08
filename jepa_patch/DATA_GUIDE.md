# Data Requirements and Flow for JEPA Patch Learning

This guide explains **what data is needed**, **where it comes from**, and **how to prepare it** for JEPA patch learning.

## TL;DR

```bash
# 1. Run VGGT-Long (existing pipeline)
python vggt_long.py --image_dir ./my_images

# 2. Save metadata (new step)
python jepa_patch/save_metadata_helper.py ./exps/my_images_*/2026-01-08-* --auto

# 3. Train patch encoder
python jepa_patch/train.py --data_dir ./exps/my_images_*/2026-01-08-*
```

---

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Your Raw Data                                            │
│    - RGB images (jpg/png)                                   │
│    - From: camera, video, dataset                          │
│    Location: ./my_images/*.png                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. VGGT-Long Processing (EXISTING PIPELINE)                │
│                                                              │
│    python vggt_long.py --image_dir ./my_images             │
│                                                              │
│    What it does:                                            │
│    ✓ Load RGB images                                       │
│    ✓ Predict depth (VGGT/Pi3/MapAnything)                 │
│    ✓ Estimate camera poses                                 │
│    ✓ Compute intrinsics                                    │
│    ✓ Split into chunks (e.g., 60 frames, 30 overlap)      │
│    ✓ Detect loop closures                                  │
│    ✓ Align chunks with SIM(3)                             │
│    ✓ Save to disk                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. VGGT-Long Output (SAVED TO DISK)                        │
│    Location: ./exps/my_images_*/YYYY-MM-DD-HH-MM-SS/       │
│                                                              │
│    _tmp_results_unaligned/                                  │
│    ├── chunk_0.npy ← Contains ALL data for chunk 0        │
│    ├── chunk_1.npy                                         │
│    └── ...                                                  │
│                                                              │
│    Each chunk_X.npy contains:                              │
│    {                                                        │
│      'depth': (N, H, W) depth maps                         │
│      'depth_conf': (N, H, W) confidence scores             │
│      'extrinsic': (N, 4, 4) camera poses (C2W)            │
│      'intrinsic': (N, 3, 3) camera intrinsics             │
│      'world_points': (N, H, W, 3) 3D points               │
│      'world_points_conf': (N, H, W) point confidence      │
│      'images': (N, H, W, 3) processed RGB                 │
│      'mask': (N, H, W) optional validity mask             │
│    }                                                        │
│                                                              │
│    loop_closures.txt (if loop detection enabled)           │
│    ├── "frame_i frame_j score"                            │
│    └── ...                                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Create Metadata (NEW STEP - YOU NEED TO DO THIS)       │
│                                                              │
│    python jepa_patch/save_metadata_helper.py \             │
│        ./exps/my_images_*/2026-01-08-* --auto              │
│                                                              │
│    Or manually:                                             │
│    python jepa_patch/save_metadata_helper.py \             │
│        ./exps/my_images_*/2026-01-08-* \                   │
│        --num_images 5000 --chunk_size 60 --overlap 30      │
│                                                              │
│    Creates: metadata.json                                  │
│    {                                                        │
│      "chunk_indices": [[0, 60], [30, 90], ...],           │
│      "loop_list": [[812, 67], [1584, 139], ...],          │
│      "num_images": 5000,                                   │
│      "chunk_size": 60,                                     │
│      "overlap": 30                                         │
│    }                                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. JEPA Patch Learning (NEW PIPELINE)                      │
│                                                              │
│    python jepa_patch/train.py \                            │
│        --data_dir ./exps/my_images_*/2026-01-08-*          │
│                                                              │
│    What it does:                                            │
│    ✓ Reads chunk_X.npy files                              │
│    ✓ Reads metadata.json                                   │
│    ✓ Samples patch centers based on geometry              │
│    ✓ Extracts patch pairs (overlap + loop closure)        │
│    ✓ Trains BYOL encoder                                   │
│    ✓ Saves checkpoints                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Trained Encoder                                         │
│    Location: ./jepa_patch_outputs/encoder_latest.pt       │
│                                                              │
│    Use for:                                                │
│    ✓ Loop closure verification                            │
│    ✓ Drift detection                                       │
│    ✓ Semantic stabilization                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Data Requirements

### What JEPA Patch Learning Needs

The patch learning system needs **per-frame data** for each image:

| Data | Shape | Description | Source |
|------|-------|-------------|--------|
| **RGB** | (H, W, 3) | Original image | Your input |
| **Depth** | (H, W) | Predicted depth (non-metric) | VGGT/Pi3/MapAnything |
| **Extrinsic** | (4, 4) | Camera pose (C2W) | VGGT-Long pose estimation |
| **Intrinsic** | (3, 3) | Camera intrinsics K | VGGT-Long (from FoV) |
| **Depth Conf** | (H, W) | Depth confidence | VGGT/Pi3/MapAnything |
| **Chunk Structure** | List[(int, int)] | Chunk boundaries | VGGT-Long chunking |
| **Loop Closures** | List[(int, int)] | Loop closure pairs | VGGT-Long loop detection |

### Where Data Is Stored

After running VGGT-Long, all data is in `output/_tmp_results_unaligned/chunk_X.npy`:

```python
import numpy as np

# Load chunk 0
chunk_data = np.load('output/_tmp_results_unaligned/chunk_0.npy', allow_pickle=True).item()

# Access data
depth = chunk_data['depth']           # (N, H, W) - N frames in chunk
extrinsics = chunk_data['extrinsic']  # (N, 4, 4)
intrinsics = chunk_data['intrinsic']  # (N, 3, 3)

# Frame i in chunk
depth_i = depth[i]
pose_i = extrinsics[i]
K_i = intrinsics[i]
```

---

## Step-by-Step: Preparing Data

### Option A: Automatic (Recommended)

```bash
# 1. Run VGGT-Long
python vggt_long.py --image_dir ./my_images

# Output will be in: ./exps/my_images_[...]/2026-01-08-[...]/

# 2. Find output directory
OUTPUT_DIR=$(ls -td exps/my_images_*/20* | head -1)
echo "Output directory: $OUTPUT_DIR"

# 3. Auto-generate metadata
python jepa_patch/save_metadata_helper.py $OUTPUT_DIR --auto

# 4. Verify metadata was created
ls $OUTPUT_DIR/metadata.json

# 5. Train patch encoder
python jepa_patch/train.py --data_dir $OUTPUT_DIR
```

### Option B: Manual (If Auto Fails)

```bash
# 1. Run VGGT-Long
python vggt_long.py --image_dir ./my_images

# 2. Count your images
NUM_IMAGES=$(ls ./my_images/*.png | wc -l)
echo "Total images: $NUM_IMAGES"

# 3. Check config for chunk settings
grep -A2 "chunk_size" configs/base_config.yaml
# Example output:
#   chunk_size: 60
#   overlap: 30

# 4. Generate metadata manually
OUTPUT_DIR=$(ls -td exps/my_images_*/20* | head -1)
python jepa_patch/save_metadata_helper.py $OUTPUT_DIR \
    --num_images $NUM_IMAGES \
    --chunk_size 60 \
    --overlap 30

# 5. Train
python jepa_patch/train.py --data_dir $OUTPUT_DIR
```

---

## Understanding the Metadata

The `metadata.json` file contains:

```json
{
  "chunk_indices": [
    [0, 60],      // Chunk 0: frames 0-59
    [30, 90],     // Chunk 1: frames 30-89 (30 frame overlap)
    [60, 120],    // Chunk 2: frames 60-119
    ...
  ],
  "loop_list": [
    [812, 67],    // Frame 812 loops back to frame 67
    [1584, 139],  // Frame 1584 loops back to frame 139
    ...
  ],
  "num_images": 5000,
  "chunk_size": 60,
  "overlap": 30
}
```

### Chunk Indices

Used to:
- Know which frames belong to which chunk
- Load the correct `chunk_X.npy` file for a given frame
- Sample overlap pairs (frames in adjacent chunks)

### Loop List

Used to:
- Sample long-horizon positive pairs for training
- Verify loop closures during evaluation
- Test the encoder on hard cases

---

## Common Issues

### Issue 1: "No chunk files found"

**Problem**: VGGT-Long didn't run successfully or output is in different location.

**Solution**:
```bash
# Find where VGGT-Long saved data
find . -name "chunk_0.npy"

# Use that directory
python jepa_patch/save_metadata_helper.py /path/to/correct/dir --auto
```

### Issue 2: "metadata.json not found"

**Problem**: Forgot to run the metadata helper.

**Solution**:
```bash
# Run metadata helper
python jepa_patch/save_metadata_helper.py ./output --auto
```

### Issue 3: "No loop closures found"

**Problem**: Loop detection was disabled or failed.

**Solution**: This is OK! You can still train with overlap pairs only:
```bash
# Training still works without loop closures
python jepa_patch/train.py --data_dir ./output

# Encoder will learn from overlap pairs (short-horizon consistency)
# Performance may be slightly lower without long-horizon examples
```

### Issue 4: Wrong number of images

**Problem**: Auto-detection got the wrong number.

**Solution**: Manually specify:
```bash
# Count your actual images
NUM_IMAGES=$(ls ./my_images/*.png | wc -l)

# Use manual mode
python jepa_patch/save_metadata_helper.py ./output \
    --num_images $NUM_IMAGES \
    --chunk_size 60 \
    --overlap 30
```

---

## Data Format Reference

### Chunk File (.npy)

```python
{
    'depth': np.ndarray,            # (N, H, W) float32
    'depth_conf': np.ndarray,       # (N, H, W) float32
    'extrinsic': np.ndarray,        # (N, 4, 4) float64 - C2W transforms
    'intrinsic': np.ndarray,        # (N, 3, 3) float64 - K matrices
    'world_points': np.ndarray,     # (N, H, W, 3) float32 - 3D points
    'world_points_conf': np.ndarray,# (N, H, W) float32
    'images': np.ndarray,           # (N, H, W, 3) uint8 - RGB
    'mask': np.ndarray or None      # (N, H, W) bool
}
```

### Metadata (.json)

```python
{
    "chunk_indices": List[List[int, int]],  # [(start, end), ...]
    "loop_list": List[List[int, int]],      # [(i, j), ...]
    "num_images": int,
    "chunk_size": int,
    "overlap": int
}
```

### Loop Closures (.txt)

```
frame_i frame_j similarity_score
812 67 0.9234
1584 139 0.8876
...
```

---

## Minimal Working Example

```bash
# Complete workflow from scratch

# 1. Extract frames from video (if needed)
mkdir ./my_video_frames
ffmpeg -i my_video.mp4 -vf "fps=5" ./my_video_frames/frame_%06d.png

# 2. Run VGGT-Long
python vggt_long.py --image_dir ./my_video_frames

# 3. Find output
OUTPUT=$(ls -td exps/my_video_frames_*/20* | head -1)

# 4. Generate metadata
python jepa_patch/save_metadata_helper.py $OUTPUT --auto

# 5. Train (quick test with 1000 steps)
python jepa_patch/train.py \
    --data_dir $OUTPUT \
    --num_steps 1000 \
    --batch_size 16

# 6. Evaluate
python jepa_patch/evaluate.py \
    --encoder_path ./jepa_patch_outputs/encoder_latest.pt \
    --data_dir $OUTPUT \
    --num_samples 100
```

---

## Summary

| Step | What | Where | Status |
|------|------|-------|--------|
| 1 | Raw images | `./my_images/*.png` | **Your data** |
| 2 | Run VGGT-Long | `python vggt_long.py --image_dir ./my_images` | **Existing pipeline** |
| 3 | Chunk files | `./exps/.../chunk_X.npy` | **Auto-generated** |
| 4 | Metadata | `python jepa_patch/save_metadata_helper.py` | **NEW - You must run** |
| 5 | Train encoder | `python jepa_patch/train.py` | **NEW - Ready to use** |

**Key Point**: The only new step you need to do is **Step 4** (generate metadata). Everything else either already exists (VGGT-Long) or is ready to use (training script).

---

## Questions?

**Q: Do I need to modify VGGT-Long code?**
A: No! Just run the metadata helper after VGGT-Long finishes.

**Q: What if I don't have loop closures?**
A: That's fine. The encoder will train on overlap pairs only.

**Q: Can I use this with Pi3 or MapAnything?**
A: Yes! The patch learning works with any VGGT-Long-compatible model.

**Q: How much disk space do I need?**
A: Same as VGGT-Long (~5-50 GiB depending on sequence length). Patch learning reads existing chunks, doesn't duplicate data.

**Q: Can I train on multiple sequences?**
A: Yes, but each sequence needs its own metadata.json. Train separately or merge datasets (advanced).
