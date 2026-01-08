"""
Example workflow: Using JEPA patch learning with VGGT-Long

This script demonstrates the complete workflow:
1. Run VGGT-Long to process a sequence
2. Save metadata for patch learning
3. Train the patch encoder
4. Evaluate the encoder
5. Use for loop closure verification

Note: This is a demonstration script showing the API.
In practice, you'd run each phase separately.
"""

import numpy as np
import json
from pathlib import Path
import torch


def example_vggt_long_with_jepa(image_dir, output_dir):
    """
    Complete example workflow.
    """
    print("="*60)
    print("JEPA Patch Learning Workflow Example")
    print("="*60)

    # ========================================
    # Phase 1: Run VGGT-Long
    # ========================================
    print("\n[1/5] Running VGGT-Long...")

    from vggt_long import VGGT_Long
    from loop_utils.config_utils import load_config

    config = load_config("configs/base_config.yaml")

    vggt_long = VGGT_Long(
        image_dir=image_dir,
        save_dir=output_dir,
        config=config
    )

    # Get image list
    import glob
    vggt_long.img_list = sorted(glob.glob(f"{image_dir}/*.png"))

    print(f"  Loaded {len(vggt_long.img_list)} images")

    # Process sequence
    print("  Processing chunks...")
    vggt_long.process_long_sequence()

    # Detect loop closures
    if vggt_long.loop_enable:
        print("  Detecting loop closures...")
        vggt_long.get_loop_pairs()

    print(f"  ✓ Processed {len(vggt_long.chunk_indices)} chunks")
    print(f"  ✓ Detected {len(vggt_long.loop_list)} loop closures")

    # ========================================
    # Phase 2: Save metadata for patch learning
    # ========================================
    print("\n[2/5] Saving metadata for patch learning...")

    metadata = {
        "chunk_indices": vggt_long.chunk_indices,
        "loop_list": vggt_long.loop_list
    }

    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"  ✓ Saved metadata to {metadata_path}")

    # ========================================
    # Phase 3: Train patch encoder
    # ========================================
    print("\n[3/5] Training patch encoder...")
    print("  (This would normally be run as a separate script)")

    # In practice, you would run:
    # python jepa_patch/train.py --data_dir ./output --num_steps 50000

    # For this example, we'll show the training setup
    from jepa_patch.patch_dataset import PatchPairDataset, collate_fn
    from jepa_patch.patch_encoder import BYOL_PatchEncoder, compute_byol_loss
    from torch.utils.data import DataLoader

    print("  Creating dataset...")
    dataset = PatchPairDataset(
        data_dir=output_dir,
        chunk_indices=metadata["chunk_indices"],
        loop_list=metadata["loop_list"],
        num_patches_per_pair=16,
        patch_size=64,
        augmentation=True
    )

    print(f"  ✓ Dataset has {len(dataset)} patch pairs")

    print("  Creating model...")
    model = BYOL_PatchEncoder(
        in_channels=5,
        encoder_hidden_dims=[64, 128, 256],
        embed_dim=256,
        ema_momentum=0.99
    )

    if torch.cuda.is_available():
        model = model.cuda()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created ({num_params:,} parameters)")

    print("\n  Training loop (showing API, not actually training):")
    print("  ---")

    loader = DataLoader(
        dataset,
        batch_size=4,  # Small batch for demo
        shuffle=True,
        collate_fn=collate_fn
    )

    # Show one training step
    model.train()
    for batch in loader:
        if not batch.get('valid', True):
            continue

        x_a = batch['patch_a']
        x_b = batch['patch_b']

        if torch.cuda.is_available():
            x_a = x_a.cuda()
            x_b = x_b.cuda()

        # Forward pass
        outputs = model(x_a, x_b)

        # Compute loss
        loss, metrics = compute_byol_loss(outputs)

        print(f"  Step 0: loss={loss.item():.4f}, pos_sim={metrics['pos_sim']:.4f}, embed_std={metrics['embed_std']:.4f}")

        # In actual training:
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # model.update_target_network()

        break  # Just one step for demo

    print("  ---")
    print("  Note: In practice, run train.py for 50k-200k steps")

    # Save encoder for demo (even though not trained)
    encoder_path = Path(output_dir) / "encoder_demo.pt"
    torch.save({
        'encoder_state_dict': model.online_encoder.state_dict(),
        'config': {
            'model': {
                'in_channels': 5,
                'encoder_hidden_dims': [64, 128, 256],
                'embed_dim': 256,
                'proj_dim': 256,
                'pred_dim': 256
            }
        }
    }, encoder_path)

    print(f"  ✓ Saved encoder to {encoder_path}")

    # ========================================
    # Phase 4: Evaluate encoder
    # ========================================
    print("\n[4/5] Evaluating encoder...")
    print("  (This would normally be run after training completes)")

    # In practice, you would run:
    # python jepa_patch/evaluate.py --encoder_path ./output/encoder.pt --data_dir ./output

    print("  Note: Evaluation requires a trained encoder")
    print("  Expected metrics after training:")
    print("    - E1 AUC: 0.85-0.90")
    print("    - E3 Verification rate: 90-95%")

    # ========================================
    # Phase 5: Use for loop closure verification
    # ========================================
    print("\n[5/5] Using for loop closure verification...")

    from jepa_patch.loop_closure_integration import LoopClosureVerifier

    print("  Creating verifier...")
    verifier = LoopClosureVerifier(
        encoder_path=encoder_path,
        threshold=0.7,
        num_patches=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("  Verifying loop closures...")

    # Example: verify first loop closure (if exists)
    if len(vggt_long.loop_list) > 0:
        frame_i, frame_j = vggt_long.loop_list[0]

        print(f"  Verifying loop closure: ({frame_i}, {frame_j})")

        # In practice, you would load frame data from chunks
        # For demo, we'll just show the API:

        # frame_i_data = {
        #     'depth': depth_i,
        #     'intrinsic': K_i,
        #     'extrinsic': T_wc_i
        # }
        #
        # frame_j_data = {
        #     'depth': depth_j,
        #     'intrinsic': K_j,
        #     'extrinsic': T_wc_j
        # }
        #
        # is_valid, score, details = verifier.verify_loop_closure(
        #     frame_i_data,
        #     frame_j_data
        # )
        #
        # if is_valid:
        #     print(f"  ✓ Valid loop closure (score={score:.4f})")
        # else:
        #     print(f"  ✗ Rejected (score={score:.4f})")

        print("  Note: Requires loading actual frame data from chunks")

    else:
        print("  Note: No loop closures detected in this sequence")

    # ========================================
    # Complete!
    # ========================================
    print("\n" + "="*60)
    print("Workflow Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train the encoder for 50k-200k steps:")
    print("   python jepa_patch/train.py --data_dir ./output --num_steps 50000")
    print("\n2. Evaluate the trained encoder:")
    print("   python jepa_patch/evaluate.py --encoder_path ./output/encoder.pt --data_dir ./output")
    print("\n3. Integrate with VGGT-Long for loop closure verification")
    print("   (See jepa_patch/loop_closure_integration.py)")


def minimal_example():
    """
    Minimal example showing just the patch encoder API.
    """
    print("\n" + "="*60)
    print("Minimal Example: Using the Patch Encoder")
    print("="*60)

    import torch
    from jepa_patch.patch_encoder import BYOL_PatchEncoder, compute_byol_loss

    # Create model
    model = BYOL_PatchEncoder(
        in_channels=5,
        encoder_hidden_dims=[64, 128, 256],
        embed_dim=256
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy batch
    B = 8  # batch size
    patch_a = torch.randn(B, 5, 64, 64)  # (B, C, H, W)
    patch_b = torch.randn(B, 5, 64, 64)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(patch_a, patch_b)

    print(f"Embedding shape: {outputs['h_a'].shape}")

    # Compute loss (for training)
    model.train()
    outputs = model(patch_a, patch_b)
    loss, metrics = compute_byol_loss(outputs)

    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Update target network (do this after each optimizer step)
    model.update_target_network()

    print("\n✓ Minimal example complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='JEPA patch learning workflow example')
    parser.add_argument('--mode', type=str, choices=['full', 'minimal'], default='minimal',
                        help='Run full workflow or minimal example')
    parser.add_argument('--image_dir', type=str, default='./images',
                        help='Directory with input images (for full workflow)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory (for full workflow)')

    args = parser.parse_args()

    if args.mode == 'minimal':
        minimal_example()
    else:
        example_vggt_long_with_jepa(args.image_dir, args.output_dir)
