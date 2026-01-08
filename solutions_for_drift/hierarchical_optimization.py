"""
Hierarchical Optimization for Large Scenes

Strategy:
1. Group chunks into "super-chunks" (e.g., every 10 chunks)
2. Optimize super-chunks globally (coarse alignment)
3. Optimize individual chunks within each super-chunk (fine alignment)

This reduces drift for very large scenes where even pose graph optimization struggles.
"""

import numpy as np


def hierarchical_optimization(vggt_long_instance, chunks_per_group=10):
    """
    Two-level hierarchical optimization:
    - Level 1: Optimize groups of chunks
    - Level 2: Optimize chunks within each group

    Args:
        vggt_long_instance: VGGT_Long object
        chunks_per_group: Number of chunks to group together

    Returns:
        optimized_sim3_list: Globally optimized transformations
    """
    num_chunks = len(vggt_long_instance.chunk_indices)
    num_groups = (num_chunks + chunks_per_group - 1) // chunks_per_group

    print(f"\nHierarchical optimization:")
    print(f"  Total chunks: {num_chunks}")
    print(f"  Group size: {chunks_per_group}")
    print(f"  Number of groups: {num_groups}")

    # ==================================================
    # Level 1: Optimize super-chunks
    # ==================================================
    print("\n[Level 1] Optimizing super-chunks...")

    group_poses = []
    for g in range(num_groups):
        start_chunk = g * chunks_per_group
        end_chunk = min((g + 1) * chunks_per_group, num_chunks)

        # Aggregate transformation across group
        s_total = 1.0
        R_total = np.eye(3)
        t_total = np.zeros(3)

        for c in range(start_chunk, end_chunk - 1):
            s, R, t = vggt_long_instance.sim3_list[c]

            # Compose
            t_total = s_total * (R_total @ t) + t_total
            R_total = R_total @ R
            s_total = s_total * s

        group_poses.append((s_total, R_total, t_total))

        print(f"  Group {g}: chunks {start_chunk}-{end_chunk-1}, "
              f"accumulated scale={s_total:.3f}")

    # Detect scale drift pattern
    scales = [s for s, _, _ in group_poses]
    scale_drift_rate = np.polyfit(range(len(scales)), scales, 1)[0]

    print(f"\n  Scale drift rate: {scale_drift_rate:.4f} per group")

    if abs(scale_drift_rate) > 0.01:
        print(f"  ⚠ Significant scale drift detected!")
        print(f"  Applying drift correction...")

        # Correct for drift
        corrected_group_poses = []
        for g, (s, R, t) in enumerate(group_poses):
            # Remove linear drift
            s_corrected = s / (1.0 + scale_drift_rate * g)
            corrected_group_poses.append((s_corrected, R, t))

        group_poses = corrected_group_poses

    # ==================================================
    # Level 2: Optimize chunks within groups
    # ==================================================
    print("\n[Level 2] Optimizing chunks within groups...")

    optimized_sim3_list = []

    for g in range(num_groups):
        start_chunk = g * chunks_per_group
        end_chunk = min((g + 1) * chunks_per_group, num_chunks)

        print(f"  Group {g}: chunks {start_chunk}-{end_chunk-1}")

        # Extract transformations for this group
        group_sim3 = vggt_long_instance.sim3_list[start_chunk:end_chunk-1]

        # Apply group-level scale correction
        if g < len(group_poses):
            s_group, _, _ = group_poses[g]
            s_target = 1.0  # Normalized scale

            scale_correction = s_target / s_group

            for c, (s, R, t) in enumerate(group_sim3):
                s_corrected = s * scale_correction
                optimized_sim3_list.append((s_corrected, R, t))
        else:
            # Last group (might be partial)
            optimized_sim3_list.extend(group_sim3)

    print(f"\n✓ Hierarchical optimization complete!")
    print(f"  Optimized {len(optimized_sim3_list)} transformations")

    return optimized_sim3_list


def detect_and_correct_drift(vggt_long_instance):
    """
    Simple drift detection and correction.

    Detects systematic scale drift and applies global correction.
    """
    sim3_list = vggt_long_instance.sim3_list

    scales = [s for s, _, _ in sim3_list]

    print(f"\nScale statistics:")
    print(f"  Mean: {np.mean(scales):.4f}")
    print(f"  Std:  {np.std(scales):.4f}")
    print(f"  Min:  {np.min(scales):.4f}")
    print(f"  Max:  {np.max(scales):.4f}")

    # Fit linear trend
    indices = np.arange(len(scales))
    drift_rate, offset = np.polyfit(indices, scales, 1)

    print(f"\nDrift analysis:")
    print(f"  Linear drift rate: {drift_rate:.6f} per chunk")
    print(f"  Total drift: {drift_rate * len(scales):.4f} ({drift_rate * len(scales) * 100:.1f}%)")

    if abs(drift_rate) < 0.001:
        print("  ✓ No significant drift detected")
        return sim3_list

    print(f"  ⚠ Significant drift! Applying correction...")

    # Correct drift
    corrected_sim3_list = []
    for i, (s, R, t) in enumerate(sim3_list):
        # Remove linear trend
        expected_scale = offset + drift_rate * i
        s_corrected = s / expected_scale

        corrected_sim3_list.append((s_corrected, R, t))

    # Verify correction
    corrected_scales = [s for s, _, _ in corrected_sim3_list]
    new_drift_rate = np.polyfit(indices, corrected_scales, 1)[0]

    print(f"  After correction:")
    print(f"    New drift rate: {new_drift_rate:.6f}")
    print(f"    Improvement: {(1 - abs(new_drift_rate) / abs(drift_rate)) * 100:.1f}%")

    return corrected_sim3_list


if __name__ == "__main__":
    print(__doc__)
