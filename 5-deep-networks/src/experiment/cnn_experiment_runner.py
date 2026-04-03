# src/experiment/cnn_experiment_runner.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: CNNExperimentRunner executes the CNN optimizer sweep.
#              Trains each DigitNetwork variant on Fashion MNIST,
#              records results, and saves a CSV log incrementally.

import time
import csv
from pathlib import Path
from typing import List

import torch
import torch.optim as optim

from src.network.digit_network          import DigitNetwork
from src.data.fashion_loader            import FashionMNISTLoader
from src.training.trainer               import Trainer
from src.evaluation.evaluator           import Evaluator
from src.experiment.cnn_experiment_config import CNNExperimentConfig, CNNExperimentResult


class CNNExperimentRunner:
    """
    Executes the CNN optimizer hyperparameter sweep.

    For each configuration:
        1. Builds a fresh DigitNetwork
        2. Constructs the specified optimizer
        3. Trains for the configured number of epochs on Fashion MNIST
        4. Evaluates on the test set
        5. Records and saves results to CSV incrementally

    Author: Krushna Sanjay Sharma
    """

    def __init__(
        self,
        device:      torch.device,
        data_dir:    str = "./data",
        output_dir:  str = "./outputs",
        results_csv: str = "task5b_cnn_results.csv",
    ):
        """
        Initialises the runner and pre-loads Fashion MNIST.

        Args:
            device      (torch.device): Target compute device.
            data_dir    (str):          Fashion MNIST cache directory.
            output_dir  (str):          Output directory for CSV.
            results_csv (str):          CSV filename for results.
        """
        self._device     = device
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path   = self._output_dir / results_csv
        self._results: List[CNNExperimentResult] = []

        print("  [CNNExperimentRunner] Loading Fashion MNIST...")
        self._data_loader = FashionMNISTLoader(
            data_dir   = data_dir,
            batch_size = CNNExperimentConfig.DEFAULT_BATCH_SIZE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, configs: list) -> List[CNNExperimentResult]:
        """
        Executes all CNN experiment configurations in order.

        Args:
            configs (list): List of config dicts from CNNExperimentConfig.

        Returns:
            List[CNNExperimentResult]: All collected results.
        """
        total = len(configs)
        print(f"\n  [CNNExperimentRunner] Starting sweep: {total} runs\n")

        self._init_csv()

        for i, cfg in enumerate(configs):
            print(f"\n  {'='*55}")
            print(f"  Run {i+1}/{total} | Phase: {cfg['phase']}")
            print(f"  optimizer={cfg['optimizer']}  lr={cfg['lr']}  "
                  f"momentum={cfg['momentum']}  wd={cfg['weight_decay']}")
            print(f"  {'='*55}")

            result = self._run_single(cfg)
            self._results.append(result)
            self._append_csv(result)

            print(f"  --> Accuracy: {result.test_accuracy:.2f}%  "
                  f"Time: {result.train_time_s:.1f}s")

        print(f"\n  [CNNExperimentRunner] Complete. Results: {self._csv_path}")
        return self._results

    def get_results(self) -> List[CNNExperimentResult]:
        """Returns all collected CNN experiment results."""
        return self._results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_optimiser(
        self,
        model: DigitNetwork,
        cfg:   dict,
    ) -> optim.Optimizer:
        """
        Constructs the optimizer specified in the config.

        Supported optimizers:
            sgd     — SGD with momentum
            adam    — Adam
            adamw   — AdamW with weight decay
            rmsprop — RMSprop

        Args:
            model (DigitNetwork): Model whose parameters to optimise.
            cfg   (dict):         Config dict with optimizer, lr, momentum,
                                  weight_decay fields.

        Returns:
            Configured torch Optimizer.

        Raises:
            ValueError: If the optimizer name is not recognised.
        """
        name = cfg["optimizer"].lower()
        lr   = cfg["lr"]
        wd   = cfg["weight_decay"]

        if name == "sgd":
            return optim.SGD(
                model.parameters(),
                lr       = lr,
                momentum = cfg["momentum"],
            )
        elif name == "adam":
            return optim.Adam(
                model.parameters(),
                lr           = lr,
                weight_decay = wd,
            )
        elif name == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr           = lr,
                weight_decay = wd,
            )
        elif name == "rmsprop":
            return optim.RMSprop(
                model.parameters(),
                lr       = lr,
                momentum = cfg["momentum"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def _run_single(self, cfg: dict) -> CNNExperimentResult:
        """
        Trains and evaluates one CNN configuration.

        Args:
            cfg (dict): Configuration dict.

        Returns:
            CNNExperimentResult with recorded metrics.
        """
        # Fresh model for each run
        model = DigitNetwork().to(self._device)

        optimiser = self._build_optimiser(model, cfg)

        trainer   = Trainer(
            model        = model,
            optimiser    = optimiser,
            device       = self._device,
            log_interval = 999,   # suppress per-batch logs during sweep
        )
        evaluator = Evaluator(model=model, device=self._device)

        # Train and time
        start = time.time()
        for epoch in range(1, cfg["epochs"] + 1):
            trainer.train_epoch(self._data_loader.train_loader, epoch_num=epoch)
        elapsed = time.time() - start

        # Evaluate on test set
        test_acc = evaluator.evaluate(self._data_loader.test_loader)

        return CNNExperimentResult(
            run_id           = cfg["run_id"],
            phase            = cfg["phase"],
            optimizer_name   = cfg["optimizer"],
            lr               = cfg["lr"],
            momentum         = cfg["momentum"],
            weight_decay     = cfg["weight_decay"],
            test_accuracy    = test_acc,
            train_time_s     = elapsed,
            epochs           = cfg["epochs"],
            final_train_loss = trainer.train_losses[-1],
            final_train_acc  = trainer.train_accuracies[-1],
        )

    def _init_csv(self) -> None:
        """Writes the CSV header row."""
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "phase", "optimizer", "lr", "momentum",
                "weight_decay", "epochs", "test_accuracy",
                "train_time_s", "final_train_loss", "final_train_acc",
            ])

    def _append_csv(self, result: CNNExperimentResult) -> None:
        """Appends one result row to the CSV (crash-safe)."""
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                result.run_id,
                result.phase,
                result.optimizer_name,
                result.lr,
                result.momentum,
                result.weight_decay,
                result.epochs,
                f"{result.test_accuracy:.4f}",
                f"{result.train_time_s:.1f}",
                f"{result.final_train_loss:.4f}",
                f"{result.final_train_acc:.2f}",
            ])