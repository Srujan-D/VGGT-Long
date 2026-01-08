"""
JEPA/BYOL-style patch encoder for learning stable 3D patch representations.

Implements:
- CNN encoder for patch tensors
- Projection and prediction heads
- EMA target network
- BYOL-style training objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class PatchEncoder(nn.Module):
    """
    CNN encoder for patch tensors.

    Treats the patch representation channels (Xn, Yn, Zn, z_rel, mask)
    as input features to a 2D CNN.
    """

    def __init__(
        self,
        in_channels=5,
        hidden_dims=[64, 128, 256],
        embed_dim=256,
        dropout=0.1
    ):
        """
        Args:
            in_channels: number of input channels (default: 5)
            hidden_dims: list of hidden layer dimensions
            embed_dim: output embedding dimension
            dropout: dropout rate
        """
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Convolutional layers
        layers = []
        prev_dim = in_channels

        for i, dim in enumerate(hidden_dims):
            layers.extend([
                nn.Conv2d(prev_dim, dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if i < len(hidden_dims) - 1 else nn.Identity()
            ])
            prev_dim = dim

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Final projection to embedding space
        self.fc = nn.Linear(hidden_dims[-1], embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) patch tensors

        Returns:
            h: (B, embed_dim) embeddings
        """
        # Convolutional layers
        features = self.conv_layers(x)  # (B, hidden_dims[-1], H', W')

        # Global pooling
        pooled = self.gap(features).squeeze(-1).squeeze(-1)  # (B, hidden_dims[-1])

        # Project to embedding space
        h = self.fc(pooled)  # (B, embed_dim)

        return h


class ProjectionHead(nn.Module):
    """
    MLP projection head for BYOL.
    """

    def __init__(
        self,
        input_dim=256,
        hidden_dim=256,
        output_dim=256,
        num_layers=2
    ):
        """
        Args:
            input_dim: input dimension
            hidden_dim: hidden layer dimension
            output_dim: output dimension
            num_layers: number of layers (default: 2)
        """
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True)
                ])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True)
                ])

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PredictionHead(nn.Module):
    """
    MLP prediction head for BYOL (online network only).
    """

    def __init__(
        self,
        input_dim=256,
        hidden_dim=256,
        output_dim=256,
        num_layers=2
    ):
        """
        Args:
            input_dim: input dimension
            hidden_dim: hidden layer dimension
            output_dim: output dimension
            num_layers: number of layers (default: 2)
        """
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True)
                ])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True)
                ])

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class BYOL_PatchEncoder(nn.Module):
    """
    Complete BYOL-style encoder for patch learning.

    Consists of:
    - Online network: encoder + projector + predictor
    - Target network: encoder + projector (EMA of online)
    """

    def __init__(
        self,
        in_channels=5,
        encoder_hidden_dims=[64, 128, 256],
        embed_dim=256,
        proj_dim=256,
        pred_dim=256,
        dropout=0.1,
        ema_momentum=0.99
    ):
        """
        Args:
            in_channels: number of input channels (default: 5)
            encoder_hidden_dims: encoder hidden dimensions
            embed_dim: encoder output dimension
            proj_dim: projector output dimension
            pred_dim: predictor output dimension
            dropout: dropout rate
            ema_momentum: EMA momentum for target network (default: 0.99)
        """
        super().__init__()

        self.ema_momentum = ema_momentum

        # Online network
        self.online_encoder = PatchEncoder(
            in_channels=in_channels,
            hidden_dims=encoder_hidden_dims,
            embed_dim=embed_dim,
            dropout=dropout
        )

        self.online_projector = ProjectionHead(
            input_dim=embed_dim,
            hidden_dim=proj_dim,
            output_dim=proj_dim,
            num_layers=2
        )

        self.predictor = PredictionHead(
            input_dim=proj_dim,
            hidden_dim=pred_dim,
            output_dim=proj_dim,
            num_layers=2
        )

        # Target network (EMA of online)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        """
        Update target network using EMA of online network.
        """
        for online_params, target_params in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_params.data = (
                self.ema_momentum * target_params.data +
                (1 - self.ema_momentum) * online_params.data
            )

        for online_params, target_params in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_params.data = (
                self.ema_momentum * target_params.data +
                (1 - self.ema_momentum) * online_params.data
            )

    def forward_online(self, x):
        """
        Forward pass through online network.

        Args:
            x: (B, C, H, W) patch tensors

        Returns:
            h: (B, embed_dim) embeddings
            z: (B, proj_dim) projections
            p: (B, proj_dim) predictions
        """
        h = self.online_encoder(x)
        z = self.online_projector(h)
        p = self.predictor(z)
        return h, z, p

    @torch.no_grad()
    def forward_target(self, x):
        """
        Forward pass through target network.

        Args:
            x: (B, C, H, W) patch tensors

        Returns:
            h: (B, embed_dim) embeddings
            z: (B, proj_dim) projections
        """
        h = self.target_encoder(x)
        z = self.target_projector(h)
        return h, z

    def forward(self, x_a, x_b):
        """
        Forward pass for a pair of patches.

        Args:
            x_a: (B, C, H, W) patches from view A
            x_b: (B, C, H, W) patches from view B

        Returns:
            dict with keys:
                - h_a, h_b: embeddings
                - z_a, z_b: projections (target)
                - p_a, p_b: predictions (online)
        """
        # Online network: A -> B
        h_a, z_a_online, p_a = self.forward_online(x_a)

        # Online network: B -> A
        h_b, z_b_online, p_b = self.forward_online(x_b)

        # Target network
        _, z_a_target = self.forward_target(x_a)
        _, z_b_target = self.forward_target(x_b)

        return {
            'h_a': h_a,
            'h_b': h_b,
            'p_a': p_a,
            'p_b': p_b,
            'z_a_target': z_a_target,
            'z_b_target': z_b_target
        }


def byol_loss(p, z):
    """
    BYOL loss: negative cosine similarity between prediction and target.

    Args:
        p: (B, D) predictions from online network
        z: (B, D) projections from target network

    Returns:
        loss: scalar loss value
    """
    p = F.normalize(p, dim=-1, p=2)
    z = F.normalize(z, dim=-1, p=2)

    loss = -2 * (p * z).sum(dim=-1).mean()

    return loss


def compute_byol_loss(outputs):
    """
    Compute symmetric BYOL loss for a batch.

    Args:
        outputs: dict from BYOL_PatchEncoder.forward()

    Returns:
        loss: scalar loss
        metrics: dict of metrics for logging
    """
    # Loss A -> B
    loss_a = byol_loss(outputs['p_a'], outputs['z_b_target'].detach())

    # Loss B -> A
    loss_b = byol_loss(outputs['p_b'], outputs['z_a_target'].detach())

    # Symmetric loss
    loss = (loss_a + loss_b) / 2

    # Compute metrics
    with torch.no_grad():
        # Positive pair similarity
        p_a_norm = F.normalize(outputs['p_a'], dim=-1, p=2)
        z_b_norm = F.normalize(outputs['z_b_target'], dim=-1, p=2)
        pos_sim_ab = (p_a_norm * z_b_norm).sum(dim=-1).mean()

        p_b_norm = F.normalize(outputs['p_b'], dim=-1, p=2)
        z_a_norm = F.normalize(outputs['z_a_target'], dim=-1, p=2)
        pos_sim_ba = (p_b_norm * z_a_norm).sum(dim=-1).mean()

        pos_sim = (pos_sim_ab + pos_sim_ba) / 2

        # Embedding standard deviation (collapse detection)
        h_a_std = outputs['h_a'].std(dim=0).mean()
        h_b_std = outputs['h_b'].std(dim=0).mean()
        embed_std = (h_a_std + h_b_std) / 2

    metrics = {
        'loss': loss.item(),
        'loss_a': loss_a.item(),
        'loss_b': loss_b.item(),
        'pos_sim': pos_sim.item(),
        'embed_std': embed_std.item()
    }

    return loss, metrics


def test_byol_encoder():
    """
    Unit test for BYOL encoder.
    """
    print("Testing BYOL encoder...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = BYOL_PatchEncoder(
        in_channels=5,
        encoder_hidden_dims=[64, 128, 256],
        embed_dim=256,
        proj_dim=256,
        pred_dim=256,
        ema_momentum=0.99
    ).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy batch
    B = 8
    C = 5
    H, W = 64, 64

    x_a = torch.randn(B, C, H, W).to(device)
    x_b = torch.randn(B, C, H, W).to(device)

    # Forward pass
    outputs = model(x_a, x_b)

    print(f"  Output shapes:")
    print(f"    h_a: {outputs['h_a'].shape}")
    print(f"    h_b: {outputs['h_b'].shape}")
    print(f"    p_a: {outputs['p_a'].shape}")
    print(f"    z_a_target: {outputs['z_a_target'].shape}")

    # Compute loss
    loss, metrics = compute_byol_loss(outputs)

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")

    # Test EMA update
    model.update_target_network()
    print("  ✓ EMA update successful")

    print("✓ BYOL encoder test passed!\n")


if __name__ == "__main__":
    test_byol_encoder()
