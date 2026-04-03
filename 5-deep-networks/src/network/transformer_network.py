# src/network/transformer_network.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Modified from source code by Prof. Bruce A. Maxwell and Andy Zhao, Spring 2026
# Description: Vision Transformer (ViT-style) network for MNIST digit
#              recognition. Drop-in replacement for DigitNetwork —
#              identical input (N, 1, 28, 28) and output (N, 10) interface.
#
# Architecture:
#   PatchEmbedding -> Dropout -> TransformerEncoder (x depth)
#   -> Token average (or CLS token) -> LayerNorm -> MLP classifier
#   -> log_softmax

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Network configuration dataclass
# ------------------------------------------------------------------

class NetConfig:
    """
    Configuration container for NetTransformer hyperparameters.

    Holds all architectural and training parameters and generates
    a CSV-formatted config string for experiment logging.

    Modified from source code by Bruce A. Maxwell and Andy Zhao,
    Spring 2026.

    Author: Krushna Sanjay Sharma
    """

    def __init__(
        self,
        name          = "vit_base",
        dataset       = "mnist",
        patch_size    = 4,
        stride        = 2,
        embed_dim     = 48,
        depth         = 4,
        num_heads     = 8,
        mlp_dim       = 128,
        dropout       = 0.1,
        use_cls_token = False,
        epochs        = 15,
        batch_size    = 64,
        lr            = 1e-3,
        weight_decay  = 1e-4,
        seed          = 0,
        optimizer     = "adamw",
        device        = "cpu",
    ):
        # Fixed dataset attributes for MNIST
        self.image_size  = 28
        self.in_channels = 1
        self.num_classes = 10

        # Variable architectural parameters
        self.name          = name
        self.dataset       = dataset
        self.patch_size    = patch_size
        self.stride        = stride
        self.embed_dim     = embed_dim
        self.depth         = depth
        self.num_heads     = num_heads
        self.mlp_dim       = mlp_dim
        self.dropout       = dropout
        self.use_cls_token = use_cls_token

        # Training parameters
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.weight_decay = weight_decay
        self.seed         = seed
        self.optimizer    = optimizer
        self.device       = device

        # CSV config string for experiment logging (TestAcc/BestEpoch filled later)
        s  = "Name,Dataset,PatchSize,Stride,Dim,Depth,Heads,MLPDim,"
        s += "Dropout,CLS,Epochs,Batch,LR,Decay,Seed,Optimizer,TestAcc,BestEpoch\n"
        s += "%s,%s,%d,%d,%d,%d,%d,%d,%.2f,%s,%d,%d,%f,%f,%d,%s," % (
            self.name, self.dataset,
            self.patch_size, self.stride,
            self.embed_dim, self.depth,
            self.num_heads, self.mlp_dim,
            self.dropout, self.use_cls_token,
            self.epochs, self.batch_size,
            self.lr, self.weight_decay,
            self.seed, self.optimizer,
        )
        self.config_string = s


# ------------------------------------------------------------------
# Patch Embedding
# ------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.

    Uses nn.Unfold to extract overlapping or non-overlapping patches,
    then projects each flattened patch into embedding space via a
    single Linear layer.

    Input:  (B, C, H, W)
    Output: (B, N, D)
        B = batch size
        N = number of patches
        D = embedding dimension

    Modified from source code by Bruce A. Maxwell and Andy Zhao,
    Spring 2026.

    Author: Krushna Sanjay Sharma
    """

    def __init__(
        self,
        image_size:  int,
        patch_size:  int,
        stride:      int,
        in_channels: int,
        embed_dim:   int,
    ) -> None:
        super().__init__()

        self.image_size  = image_size
        self.patch_size  = patch_size
        self.stride      = stride
        self.in_channels = in_channels
        self.embed_dim   = embed_dim

        # Unfold extracts sliding local blocks (patches) from the image.
        # stride == patch_size → non-overlapping
        # stride <  patch_size → overlapping
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

        # Flattened size of one patch
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projection from patch space to embedding space
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

        # Precompute number of patches for positional embedding sizing
        self.num_patches = self._compute_num_patches()

    def _compute_num_patches(self) -> int:
        """
        Computes the total number of patches extracted from one image.

        Positions along one spatial dimension:
            ((image_size - patch_size) // stride) + 1

        Total patches (square image, square patch):
            positions_per_dim ^ 2
        """
        positions = ((self.image_size - self.patch_size) // self.stride) + 1
        return positions * positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts patches and projects them to embedding vectors.

        Args:
            x (Tensor): Input images (B, C, H, W).

        Returns:
            Tensor: Patch embeddings (B, N, embed_dim).
        """
        # Step 1: extract patches → (B, patch_dim, N)
        x = self.unfold(x)

        # Step 2: transpose → (B, N, patch_dim)
        x = x.transpose(1, 2)

        # Step 3: project each patch → (B, N, embed_dim)
        x = self.proj(x)

        return x


# ------------------------------------------------------------------
# Transformer Network
# ------------------------------------------------------------------

class NetTransformer(nn.Module):
    """
    Vision Transformer for MNIST digit classification.

    Drop-in replacement for DigitNetwork — same input/output interface:
        Input:  (N, 1, 28, 28)
        Output: (N, 10) log-probabilities

    Architecture:
        PatchEmbedding
        -> Positional embedding + Dropout
        -> TransformerEncoder (depth x TransformerEncoderLayer)
        -> Token aggregation (mean pooling or CLS token)
        -> LayerNorm
        -> MLP classifier (Linear -> GELU -> Linear)
        -> log_softmax

    Modified from source code by Bruce A. Maxwell and Andy Zhao,
    Spring 2026.

    Author: Krushna Sanjay Sharma
    """

    def __init__(self, config: NetConfig):
        """
        Initialises all layers of the transformer network.

        Args:
            config (NetConfig): Hyperparameter configuration object.
        """
        super(NetTransformer, self).__init__()

        # Patch embedding: image → sequence of token embeddings
        self.patch_embed = PatchEmbedding(
            image_size  = config.image_size,
            patch_size  = config.patch_size,
            stride      = config.stride,
            in_channels = config.in_channels,
            embed_dim   = config.embed_dim,
        )

        num_tokens = self.patch_embed.num_patches
        print(f"  [NetTransformer] Number of patch tokens: {num_tokens}")

        # Optional CLS token — prepended to the token sequence
        self.use_cls_token = config.use_cls_token
        if self.use_cls_token:
            self.cls_token  = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens    = num_tokens + 1
        else:
            self.cls_token = None
            total_tokens   = num_tokens

        # Learnable positional embedding — one vector per token
        self.pos_embed   = nn.Parameter(torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)

        # Transformer encoder: stack of TransformerEncoderLayer blocks
        # Each layer includes: multi-head self-attention, FFN,
        # layer norm, residual connections, and dropout.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = config.embed_dim,
            nhead          = config.num_heads,
            dim_feedforward = config.mlp_dim,
            dropout        = config.dropout,
            activation     = "gelu",
            batch_first    = True,   # expects (B, N, D) not (N, B, D)
            norm_first     = True,   # pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers    = config.depth,
        )

        # Final layer normalisation before classification
        self.norm = nn.LayerNorm(config.embed_dim)

        # MLP classification head:
        #   Linear(embed_dim -> mlp_dim) -> GELU -> Linear(mlp_dim -> num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.num_classes),
        )

        # Initialise positional embedding and CLS token
        self._init_parameters()

    def _init_parameters(self) -> None:
        """
        Initialises positional embedding and optional CLS token with
        truncated normal distribution (std=0.02) for stable training.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass through the transformer network.

        Processing pipeline:
            patch_embed
            -> (optional CLS token prepend)
            -> positional embedding + dropout
            -> transformer encoder
            -> token aggregation (mean or CLS)
            -> layer norm
            -> MLP classifier
            -> log_softmax

        Args:
            x (Tensor): Input images of shape (B, 1, 28, 28).

        Returns:
            Tensor: Log-probabilities of shape (B, 10).
        """
        # Step 1: convert image patches to token embeddings → (B, N, D)
        x = self.patch_embed(x)

        batch_size = x.size(0)

        # Step 2: optionally prepend CLS token → (B, N+1, D)
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # Step 3: add learnable positional embedding and apply dropout
        x = self.pos_dropout(x + self.pos_embed)

        # Step 4: run through transformer encoder → (B, N, D)
        x = self.encoder(x)

        # Step 5: aggregate tokens into a single vector
        if self.use_cls_token:
            x = x[:, 0]        # use CLS token (first token)
        else:
            x = x.mean(dim=1)  # global average pooling over all tokens

        # Step 6: final layer normalisation
        x = self.norm(x)

        # Step 7: MLP classification head → (B, num_classes)
        x = self.classifier(x)

        # Step 8: log_softmax for NLL loss compatibility
        return F.log_softmax(x, dim=1)