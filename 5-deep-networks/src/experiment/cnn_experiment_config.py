# src/experiment/cnn_experiment_config.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: CNNExperimentConfig defines the hyperparameter sweep space
#              for the CNN optimizer experiment (Task 5B).
#
# Three dimensions:
#   1. optimizer   — SGD, Adam, AdamW, RMSprop
#   2. lr          — [0.1, 0.01, 0.001, 0.0001]
#   3. momentum    — [0.0, 0.5, 0.9, 0.99]  (SGD only; fixed for Adam/AdamW)
#
# Strategy: round-robin linear search
#   Phase 0 — Baseline:       SGD, lr=0.01, momentum=0.5  (matches Task 1)
#   Phase 1 — Optimizer sweep: vary optimizer, fix lr=0.01
#   Phase 2 — LR sweep:        vary lr, fix best optimizer
#   Phase 3 — Momentum sweep:  vary momentum (SGD only), fix best lr
#   Phase 4 — Cross grid:      optimizer x lr combinations

from dataclasses import dataclass
from typing import List


@dataclass
class CNNExperimentResult:
    """
    Stores the result of a single CNN training run.

    Author: Krushna Sanjay Sharma
    """
    run_id:           int
    phase:            str
    optimizer_name:   str
    lr:               float
    momentum:         float
    weight_decay:     float
    test_accuracy:    float
    train_time_s:     float
    epochs:           int
    final_train_loss: float
    final_train_acc:  float


class CNNExperimentConfig:
    """
    Defines the CNN optimizer sweep space and generates all configurations.

    Uses a round-robin linear search strategy — hold two dimensions fixed,
    sweep the third, then rotate.

    Author: Krushna Sanjay Sharma
    """

    # ------------------------------------------------------------------
    # Baseline (matches Task 1 training config)
    # ------------------------------------------------------------------
    DEFAULT_OPTIMIZER  = "sgd"
    DEFAULT_LR         = 0.01
    DEFAULT_MOMENTUM   = 0.5
    DEFAULT_WEIGHT_DECAY = 0.0
    DEFAULT_EPOCHS     = 5
    DEFAULT_BATCH_SIZE = 64

    # ------------------------------------------------------------------
    # Sweep ranges
    # ------------------------------------------------------------------
    OPTIMIZERS   = ["sgd", "adam", "adamw", "rmsprop"]
    LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]
    MOMENTUMS    = [0.0, 0.5, 0.9, 0.99]    # SGD only
    WEIGHT_DECAYS = [0.0, 1e-4, 1e-3]        # for Adam/AdamW

    def generate_sweep_configs(self) -> List[dict]:
        """
        Generates all CNN experiment configurations in phase order.

        Returns:
            List of config dicts with keys:
                run_id, phase, optimizer, lr, momentum,
                weight_decay, epochs, batch_size
        """
        configs = []
        run_id  = 0

        def make(phase, opt, lr, momentum=0.0, weight_decay=0.0):
            nonlocal run_id
            cfg = dict(
                run_id       = run_id,
                phase        = phase,
                optimizer    = opt,
                lr           = lr,
                momentum     = momentum,
                weight_decay = weight_decay,
                epochs       = self.DEFAULT_EPOCHS,
                batch_size   = self.DEFAULT_BATCH_SIZE,
            )
            run_id += 1
            return cfg

        # --- Phase 0: Baseline (SGD, lr=0.01, momentum=0.5) ---
        configs.append(make("baseline", "sgd", 0.01, momentum=0.5))

        # --- Phase 1: Optimizer sweep (fix lr=0.01) ---
        for opt in self.OPTIMIZERS:
            if opt != "sgd":   # baseline already covers sgd
                mom = 0.5 if opt == "sgd" else 0.0
                configs.append(make("optimizer_sweep", opt, 0.01, momentum=mom))

        # --- Phase 2: LR sweep for each optimizer (fix momentum=default) ---
        for opt in self.OPTIMIZERS:
            for lr in self.LEARNING_RATES:
                if opt == "sgd" and lr == 0.01:
                    continue   # baseline covers this
                mom = 0.5 if opt == "sgd" else 0.0
                configs.append(make("lr_sweep", opt, lr, momentum=mom))

        # --- Phase 3: Momentum sweep for SGD only ---
        for mom in self.MOMENTUMS:
            if mom != 0.5:     # baseline covers 0.5
                configs.append(make("momentum_sweep", "sgd", 0.01, momentum=mom))

        # --- Phase 4: Weight decay sweep for Adam/AdamW ---
        for opt in ["adam", "adamw"]:
            for wd in self.WEIGHT_DECAYS:
                if wd != 0.0:   # already covered in lr_sweep
                    configs.append(make(
                        "weight_decay_sweep", opt, 0.001,
                        momentum=0.0, weight_decay=wd,
                    ))

        print(f"  [CNNExperimentConfig] Generated {len(configs)} configurations.")
        return configs