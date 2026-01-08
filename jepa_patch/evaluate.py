"""
Evaluation utilities for JEPA patch encoder.

Implements evaluation metrics:
- E1: Same-place vs different-place separation (AUC)
- E2: Long-horizon consistency curve
- E3: Loop closure verification utility
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from patch_encoder import BYOL_PatchEncoder
from patch_dataset import PatchPairDataset, collate_fn


class PatchEncoderEvaluator:
    """
    Evaluator for trained patch encoder.
    """

    def __init__(self, encoder_path, config, device='cuda'):
        """
        Args:
            encoder_path: path to saved encoder checkpoint
            config: config dict
            device: device to use
        """
        self.device = torch.device(device)
        self.config = config

        # Load encoder
        checkpoint = torch.load(encoder_path, map_location=self.device)

        self.encoder = BYOL_PatchEncoder(
            in_channels=config['model']['in_channels'],
            encoder_hidden_dims=config['model']['encoder_hidden_dims'],
            embed_dim=config['model']['embed_dim'],
            proj_dim=config['model']['proj_dim'],
            pred_dim=config['model']['pred_dim']
        ).to(self.device)

        if 'encoder_state_dict' in checkpoint:
            self.encoder.online_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint['model_state_dict'])

        self.encoder.eval()

    @torch.no_grad()
    def encode_patches(self, patches):
        """
        Encode a batch of patches.

        Args:
            patches: (B, C, H, W) patch tensors

        Returns:
            embeddings: (B, D) embeddings
        """
        patches = patches.to(self.device)
        embeddings = self.encoder.online_encoder(patches)
        return embeddings

    @torch.no_grad()
    def compute_similarity(self, emb_a, emb_b):
        """
        Compute cosine similarity between embeddings.

        Args:
            emb_a: (N, D) or (D,) embeddings
            emb_b: (M, D) or (D,) embeddings

        Returns:
            similarity: (N, M) or scalar similarity matrix
        """
        emb_a = F.normalize(emb_a, dim=-1, p=2)
        emb_b = F.normalize(emb_b, dim=-1, p=2)

        if emb_a.ndim == 1:
            emb_a = emb_a.unsqueeze(0)
        if emb_b.ndim == 1:
            emb_b = emb_b.unsqueeze(0)

        similarity = torch.mm(emb_a, emb_b.T)

        return similarity.squeeze()

    def evaluate_same_vs_different(self, dataset, num_samples=1000):
        """
        E1: Evaluate same-place vs different-place separation.

        Computes AUC for classifying positive vs negative pairs by cosine distance.

        Args:
            dataset: PatchPairDataset
            num_samples: number of samples to evaluate

        Returns:
            metrics: dict with AUC and other metrics
        """
        print("Evaluating same-place vs different-place separation...")

        # Sample positive pairs (from dataset)
        positive_sims = []

        for i in tqdm(range(min(num_samples, len(dataset))), desc='Positive pairs'):
            sample = dataset[i]

            if not sample.get('valid', True):
                continue

            patches_a = sample['patch_a']  # (N, C, H, W)
            patches_b = sample['patch_b']

            # Encode
            emb_a = self.encode_patches(patches_a)
            emb_b = self.encode_patches(patches_b)

            # Compute pairwise similarities
            for j in range(len(emb_a)):
                sim = self.compute_similarity(emb_a[j], emb_b[j]).item()
                positive_sims.append(sim)

        # Sample negative pairs (random patches from different locations)
        negative_sims = []

        for i in tqdm(range(min(num_samples, len(dataset))), desc='Negative pairs'):
            sample_a = dataset[i]
            sample_b = dataset[(i + len(dataset) // 2) % len(dataset)]  # Far apart

            if not sample_a.get('valid', True) or not sample_b.get('valid', True):
                continue

            patches_a = sample_a['patch_a']
            patches_b = sample_b['patch_b']

            emb_a = self.encode_patches(patches_a)
            emb_b = self.encode_patches(patches_b)

            # Random pairs
            for j in range(min(len(emb_a), len(emb_b))):
                sim = self.compute_similarity(emb_a[j], emb_b[j]).item()
                negative_sims.append(sim)

        # Compute AUC
        y_true = [1] * len(positive_sims) + [0] * len(negative_sims)
        y_scores = positive_sims + negative_sims

        auc = roc_auc_score(y_true, y_scores)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        # Optimal threshold (maximize TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        metrics = {
            'auc': auc,
            'positive_mean': np.mean(positive_sims),
            'positive_std': np.std(positive_sims),
            'negative_mean': np.mean(negative_sims),
            'negative_std': np.std(negative_sims),
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx]
        }

        print(f"  AUC: {auc:.4f}")
        print(f"  Positive sim: {metrics['positive_mean']:.4f} ± {metrics['positive_std']:.4f}")
        print(f"  Negative sim: {metrics['negative_mean']:.4f} ± {metrics['negative_std']:.4f}")
        print(f"  Optimal threshold: {optimal_threshold:.4f} (TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f})")

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], c='red', s=100, zorder=5,
                    label=f'Optimal (thresh={optimal_threshold:.3f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve: Same-place vs Different-place')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(positive_sims, bins=50, alpha=0.5, label='Positive pairs', color='green')
        plt.hist(negative_sims, bins=50, alpha=0.5, label='Negative pairs', color='red')
        plt.axvline(optimal_threshold, color='blue', linestyle='--', label='Optimal threshold')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.title('Similarity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('similarity_histogram.png', dpi=150, bbox_inches='tight')
        plt.close()

        return metrics

    def evaluate_loop_closure_verification(self, dataset, threshold=None):
        """
        E3: Evaluate loop closure verification utility.

        Computes how well the patch embedding can verify loop closures.

        Args:
            dataset: PatchPairDataset
            threshold: similarity threshold (if None, use optimal from E1)

        Returns:
            metrics: dict with verification accuracy metrics
        """
        print("Evaluating loop closure verification...")

        # Filter loop closure pairs from dataset
        loop_pairs = [p for p in dataset.pairs if p['type'] == 'loop']

        if len(loop_pairs) == 0:
            print("  No loop closure pairs found in dataset!")
            return {}

        print(f"  Evaluating {len(loop_pairs)} loop closure pairs")

        similarities = []

        for pair_info in tqdm(loop_pairs, desc='Loop closures'):
            try:
                # Load frame data and extract patches
                # This is a simplified version - in practice you'd use the dataset's __getitem__
                idx = dataset.pairs.index(pair_info)
                sample = dataset[idx]

                if not sample.get('valid', True):
                    continue

                patches_a = sample['patch_a']
                patches_b = sample['patch_b']

                emb_a = self.encode_patches(patches_a)
                emb_b = self.encode_patches(patches_b)

                # Compute mean similarity across patches
                sims = []
                for i in range(min(len(emb_a), len(emb_b))):
                    sim = self.compute_similarity(emb_a[i], emb_b[i]).item()
                    sims.append(sim)

                mean_sim = np.mean(sims)
                similarities.append(mean_sim)

            except Exception as e:
                print(f"  Warning: Failed to process pair {pair_info}: {e}")
                continue

        if len(similarities) == 0:
            print("  No valid loop closure pairs processed!")
            return {}

        metrics = {
            'num_loop_closures': len(similarities),
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }

        if threshold is not None:
            verified = np.sum(np.array(similarities) >= threshold)
            metrics['verification_rate'] = verified / len(similarities)
            print(f"  Verification rate (thresh={threshold:.3f}): {metrics['verification_rate']:.2%}")

        print(f"  Mean similarity: {metrics['mean_similarity']:.4f} ± {metrics['std_similarity']:.4f}")

        return metrics

    def save_results(self, results, output_path):
        """
        Save evaluation results to JSON.

        Args:
            results: dict of evaluation results
            output_path: path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate JEPA patch encoder')
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to saved encoder checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to VGGT-Long output directory')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config from encoder checkpoint
    checkpoint = torch.load(args.encoder_path, map_location='cpu')
    config = checkpoint.get('config', {})

    # Create evaluator
    evaluator = PatchEncoderEvaluator(args.encoder_path, config, device=args.device)

    # Load dataset
    data_dir = Path(args.data_dir)
    metadata_file = data_dir / 'metadata.json'

    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found!")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    chunk_indices = metadata.get('chunk_indices', [])
    loop_list = metadata.get('loop_list', [])

    dataset = PatchPairDataset(
        data_dir=data_dir,
        chunk_indices=chunk_indices,
        loop_list=loop_list,
        num_patches_per_pair=16,
        patch_size=64,
        augmentation=False  # No augmentation for evaluation
    )

    # Run evaluations
    results = {}

    # E1: Same vs different
    print("\n" + "="*60)
    print("E1: Same-place vs Different-place Separation")
    print("="*60)
    results['same_vs_different'] = evaluator.evaluate_same_vs_different(
        dataset,
        num_samples=args.num_samples
    )

    # E3: Loop closure verification
    print("\n" + "="*60)
    print("E3: Loop Closure Verification")
    print("="*60)
    optimal_threshold = results['same_vs_different']['optimal_threshold']
    results['loop_closure_verification'] = evaluator.evaluate_loop_closure_verification(
        dataset,
        threshold=optimal_threshold
    )

    # Save results
    results_path = output_dir / 'evaluation_results.json'
    evaluator.save_results(results, results_path)

    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
