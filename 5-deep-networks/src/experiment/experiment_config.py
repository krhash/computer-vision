# src/experiment/experiment_config.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: ExperimentConfig defines the hyperparameter sweep space
#              for Task 5. Uses a round-robin linear search strategy:
#              hold two dimensions constant, sweep the third, then rotate.
#
# Three dimensions:
#   1. patch_size  — controls token granularity
#   2. embed_dim   — controls model capacity
#   3. depth       — controls number of transformer layers

from dataclasses import dataclass, field
from typing import List, Any


# ------------------------------------------------------------------
# Single experiment result container
# ------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """
    Stores the result of a single training run.

    Author: Krushna Sanjay Sharma
    """
    run_id:        int
    phase:         str          # e.g. "baseline", "patch_sweep", "embed_sweep"
    patch_size:    int
    embed_dim:     int
    depth:         int
    test_accuracy: float        # best test accuracy achieved (%)
    train_time_s:  float        # total training time in seconds
    epochs:        int
    final_loss:    float


# ------------------------------------------------------------------
# Sweep space definition
# ------------------------------------------------------------------

class ExperimentConfig:
    """
    Defines the hyperparameter sweep space and generates all experiment
    configurations using a round-robin linear search strategy.

    Strategy:
        Phase 0 — Baseline: default config
        Phase 1 — Patch sweep:  vary patch_size, fix embed/depth at default
        Phase 2 — Embed sweep:  vary embed_dim,  fix patch/depth at best
        Phase 3 — Depth sweep:  vary depth,      fix patch/embed at best
        Phase 4 — Round 2:      repeat phases 1-3 with updated bests
        Phase 5 — Top combos:   random combinations of top performers

    Target: 50-100 total runs.

    Author: Krushna Sanjay Sharma
    """

    # ------------------------------------------------------------------
    # Default (baseline) configuration
    # ------------------------------------------------------------------
    DEFAULT_PATCH_SIZE = 4
    DEFAULT_EMBED_DIM  = 48
    DEFAULT_DEPTH      = 4
    DEFAULT_EPOCHS     = 10     # reduced from 15 for sweep speed
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_LR         = 1e-3
    DEFAULT_WEIGHT_DECAY = 1e-4

    # ------------------------------------------------------------------
    # Sweep ranges per dimension
    # ------------------------------------------------------------------
    PATCH_SIZES = [2, 4, 7, 14]
    EMBED_DIMS  = [16, 32, 48, 96]
    DEPTHS      = [1, 2, 4, 6]

    def __init__(self):
        """Initialises the config with default bests (updated as sweep runs)."""
        # These are updated after each phase based on best result found
        self.best_patch = self.DEFAULT_PATCH_SIZE
        self.best_embed = self.DEFAULT_EMBED_DIM
        self.best_depth = self.DEFAULT_DEPTH

    def generate_sweep_configs(self) -> List[dict]:
        """
        Generates all experiment configurations in phase order.

        Returns:
            List of dicts, each with keys:
                run_id, phase, patch_size, embed_dim, depth,
                epochs, batch_size, lr, weight_decay
        """
        configs = []
        run_id  = 0

        def make(phase, patch, embed, depth):
            nonlocal run_id
            cfg = dict(
                run_id       = run_id,
                phase        = phase,
                patch_size   = patch,
                embed_dim    = embed,
                depth        = depth,
                epochs       = self.DEFAULT_EPOCHS,
                batch_size   = self.DEFAULT_BATCH_SIZE,
                lr           = self.DEFAULT_LR,
                weight_decay = self.DEFAULT_WEIGHT_DECAY,
            )
            run_id += 1
            return cfg

        # --- Phase 0: Baseline ---
        configs.append(make(
            "baseline",
            self.DEFAULT_PATCH_SIZE,
            self.DEFAULT_EMBED_DIM,
            self.DEFAULT_DEPTH,
        ))

        # --- Phase 1: Patch sweep (round 1) ---
        for p in self.PATCH_SIZES:
            if p != self.DEFAULT_PATCH_SIZE:   # baseline already covers default
                configs.append(make("patch_sweep_r1", p, self.DEFAULT_EMBED_DIM, self.DEFAULT_DEPTH))

        # --- Phase 2: Embed sweep (round 1) ---
        for e in self.EMBED_DIMS:
            if e != self.DEFAULT_EMBED_DIM:
                configs.append(make("embed_sweep_r1", self.DEFAULT_PATCH_SIZE, e, self.DEFAULT_DEPTH))

        # --- Phase 3: Depth sweep (round 1) ---
        for d in self.DEPTHS:
            if d != self.DEFAULT_DEPTH:
                configs.append(make("depth_sweep_r1", self.DEFAULT_PATCH_SIZE, self.DEFAULT_EMBED_DIM, d))

        # --- Phase 4: Cross sweeps — best-so-far combinations ---
        # patch × embed grid (fix depth=default)
        for p in self.PATCH_SIZES:
            for e in self.EMBED_DIMS:
                if not (p == self.DEFAULT_PATCH_SIZE and e == self.DEFAULT_EMBED_DIM):
                    configs.append(make("patch_embed_grid", p, e, self.DEFAULT_DEPTH))

        # patch × depth grid (fix embed=default)
        for p in self.PATCH_SIZES:
            for d in self.DEPTHS:
                if not (p == self.DEFAULT_PATCH_SIZE and d == self.DEFAULT_DEPTH):
                    configs.append(make("patch_depth_grid", p, self.DEFAULT_EMBED_DIM, d))

        # embed × depth grid (fix patch=default)
        for e in self.EMBED_DIMS:
            for d in self.DEPTHS:
                if not (e == self.DEFAULT_EMBED_DIM and d == self.DEFAULT_DEPTH):
                    configs.append(make("embed_depth_grid", self.DEFAULT_PATCH_SIZE, e, d))

        print(f"  [ExperimentConfig] Generated {len(configs)} configurations.")
        return configs