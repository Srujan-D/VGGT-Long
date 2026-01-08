"""
Training script for JEPA/BYOL patch encoder.

Trains the patch encoder on VGGT-Long processed data.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from patch_encoder import BYOL_PatchEncoder, compute_byol_loss
from patch_dataset import PatchPairDataset, collate_fn


class Trainer:
    """
    Trainer for BYOL patch encoder.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        # Create model
        self.model = BYOL_PatchEncoder(
            in_channels=config['model']['in_channels'],
            encoder_hidden_dims=config['model']['encoder_hidden_dims'],
            embed_dim=config['model']['embed_dim'],
            proj_dim=config['model']['proj_dim'],
            pred_dim=config['model']['pred_dim'],
            dropout=config['model']['dropout'],
            ema_momentum=config['training']['ema_momentum']
        ).to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            [
                {'params': self.model.predictor.parameters(), 'lr': config['training']['lr_head']},
                {'params': list(self.model.online_encoder.parameters()) +
                           list(self.model.online_projector.parameters()),
                 'lr': config['training']['lr_encoder']}
            ],
            weight_decay=config['training']['weight_decay']
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_steps'],
            eta_min=config['training']['lr_min']
        )

        # Metrics tracking
        self.step = 0
        self.metrics_history = {
            'loss': [],
            'pos_sim': [],
            'embed_std': [],
            'lr': []
        }

        # Output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def train_step(self, batch):
        """
        Single training step.

        Args:
            batch: dict from DataLoader

        Returns:
            metrics: dict of metrics
        """
        if not batch.get('valid', True):
            return None

        # Move to device
        x_a = batch['patch_a'].to(self.device)
        x_b = batch['patch_b'].to(self.device)

        # Forward pass
        outputs = self.model(x_a, x_b)

        # Compute loss
        loss, metrics = compute_byol_loss(outputs)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip']
            )

        self.optimizer.step()
        self.scheduler.step()

        # Update target network
        self.model.update_target_network()

        # Add learning rate to metrics
        metrics['lr'] = self.optimizer.param_groups[0]['lr']

        return metrics

    def train(self, train_loader):
        """
        Main training loop.

        Args:
            train_loader: DataLoader for training data
        """
        self.model.train()

        pbar = tqdm(total=self.config['training']['num_steps'], desc='Training')

        while self.step < self.config['training']['num_steps']:
            for batch in train_loader:
                metrics = self.train_step(batch)

                if metrics is not None:
                    # Update metrics history
                    for key, value in metrics.items():
                        if key in self.metrics_history:
                            self.metrics_history[key].append(value)

                    # Log
                    if self.step % self.config['training']['log_interval'] == 0:
                        pbar.set_postfix(metrics)
                        tqdm.write(f"Step {self.step}: {metrics}")

                    # Save checkpoint
                    if self.step % self.config['training']['save_interval'] == 0:
                        self.save_checkpoint()

                    # Plot metrics
                    if self.step % self.config['training']['plot_interval'] == 0:
                        self.plot_metrics()

                    self.step += 1
                    pbar.update(1)

                    if self.step >= self.config['training']['num_steps']:
                        break

        pbar.close()

        # Final save
        self.save_checkpoint()
        self.plot_metrics()

        print(f"\nTraining completed! Saved to {self.output_dir}")

    def save_checkpoint(self):
        """
        Save model checkpoint.
        """
        checkpoint_path = self.output_dir / f'checkpoint_step_{self.step}.pt'

        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics_history': self.metrics_history,
            'config': self.config
        }, checkpoint_path)

        # Also save latest
        latest_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics_history': self.metrics_history,
            'config': self.config
        }, latest_path)

        # Save encoder only (for inference)
        encoder_path = self.output_dir / 'encoder_latest.pt'
        torch.save({
            'encoder_state_dict': self.model.online_encoder.state_dict(),
            'config': self.config
        }, encoder_path)

        print(f"Saved checkpoint at step {self.step}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.metrics_history = checkpoint['metrics_history']

        print(f"Loaded checkpoint from step {self.step}")

    def plot_metrics(self):
        """
        Plot training metrics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss
        if len(self.metrics_history['loss']) > 0:
            axes[0, 0].plot(self.metrics_history['loss'])
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)

        # Positive similarity
        if len(self.metrics_history['pos_sim']) > 0:
            axes[0, 1].plot(self.metrics_history['pos_sim'])
            axes[0, 1].set_title('Positive Similarity')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Cosine Similarity')
            axes[0, 1].grid(True)

        # Embedding std (collapse detection)
        if len(self.metrics_history['embed_std']) > 0:
            axes[1, 0].plot(self.metrics_history['embed_std'])
            axes[1, 0].set_title('Embedding Std (Collapse Detection)')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Std')
            axes[1, 0].grid(True)

        # Learning rate
        if len(self.metrics_history['lr']) > 0:
            axes[1, 1].plot(self.metrics_history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train JEPA patch encoder')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to VGGT-Long output directory')
    parser.add_argument('--output_dir', type=str, default='./jepa_patch_outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--num_steps', type=int, default=50000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (number of patch pairs)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
    parser.add_argument('--lr_encoder', type=float, default=3e-4,
                        help='Learning rate for encoder')
    parser.add_argument('--lr_head', type=float, default=1e-3,
                        help='Learning rate for predictor head')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Configuration
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'device': args.device,
        'model': {
            'in_channels': 5,
            'encoder_hidden_dims': [64, 128, 256],
            'embed_dim': 256,
            'proj_dim': 256,
            'pred_dim': 256,
            'dropout': 0.1
        },
        'training': {
            'num_steps': args.num_steps,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'lr_encoder': args.lr_encoder,
            'lr_head': args.lr_head,
            'lr_min': 1e-6,
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'ema_momentum': 0.99,
            'log_interval': 100,
            'save_interval': 5000,
            'plot_interval': 1000
        },
        'dataset': {
            'num_patches_per_pair': 16,
            'patch_size': 64,
            'overlap_weight': 0.5,
            'loop_weight': 0.5,
            'augmentation': True
        }
    }

    # Load chunk indices and loop list from VGGT-Long output
    # TODO: These should be saved by VGGT-Long; for now we'll create a loader
    print(f"Loading data from {args.data_dir}...")
    print("Note: You need to first run VGGT-Long to generate processed chunks.")
    print("The dataset will look for _tmp_results_unaligned/chunk_*.npy files.")

    # For now, create a simple dataset
    # In practice, you'd load the actual chunk_indices and loop_list from VGGT-Long
    data_dir = Path(args.data_dir)

    # Check if data exists
    unaligned_dir = data_dir / '_tmp_results_unaligned'
    if not unaligned_dir.exists():
        print(f"Error: {unaligned_dir} not found!")
        print("Please run VGGT-Long first to process your data.")
        return

    # Load metadata if available
    metadata_file = data_dir / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        chunk_indices = metadata.get('chunk_indices', [])
        loop_list = metadata.get('loop_list', [])
    else:
        print("Warning: metadata.json not found. You need to save chunk_indices and loop_list.")
        print("For now, please create metadata.json with:")
        print("  {\"chunk_indices\": [[start, end], ...], \"loop_list\": [[i, j], ...]}")
        return

    print(f"Loaded {len(chunk_indices)} chunks, {len(loop_list)} loop closures")

    # Create dataset
    train_dataset = PatchPairDataset(
        data_dir=data_dir,
        chunk_indices=chunk_indices,
        loop_list=loop_list,
        num_patches_per_pair=config['dataset']['num_patches_per_pair'],
        patch_size=config['dataset']['patch_size'],
        overlap_weight=config['dataset']['overlap_weight'],
        loop_weight=config['dataset']['loop_weight'],
        augmentation=config['dataset']['augmentation']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create trainer
    trainer = Trainer(config)

    # Resume if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    trainer.train(train_loader)


if __name__ == "__main__":
    main()
