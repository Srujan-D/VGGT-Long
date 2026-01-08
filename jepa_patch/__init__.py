"""
JEPA-Style Pose-Normalized 3D Patch Learning for VGGT-Long

This module implements a self-supervised learning system for learning
stable 3D patch representations from RGB-D sequences with non-metric depth.

Key components:
- Pose normalization and validation (pose_utils)
- Patch representation with scale invariance (patch_representation)
- Geometric importance sampling (patch_sampling)
- Patch pair dataset builder (patch_dataset)
- JEPA/BYOL encoder (patch_encoder)
- Training and evaluation utilities
"""

__version__ = "0.1.0"
