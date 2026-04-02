"""Training logger: W&B + CSV behind a single interface."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    """
    Logs training metrics to CSV (always) and W&B (optional).

    Parameters
    ----------
    run_name      : experiment name used for W&B run and log filename
    output_dir    : directory where train_log.csv is written
    use_wandb     : whether to attempt W&B logging
    wandb_project : W&B project name (default: "drone-rl")
    cfg           : full training config dict, logged as W&B run config
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        cfg: Optional[Dict] = None,
    ) -> None:
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project or "drone-rl",
                    name=run_name,
                    config=cfg or {},
                )
                self._wandb = wandb
            except Exception:
                pass  # no API key or wandb not installed — silent skip

        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path   = log_dir / "train_log.csv"
        self._csv_file   = open(self._csv_path, "w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list] = None

    def log(self, episode: int, step: int, metrics: Dict[str, Any]) -> None:
        """Write one row to CSV (and W&B if enabled)."""
        row = {"episode": episode, "step": step, **metrics}

        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def close(self) -> None:
        """Flush and close all backends."""
        self._csv_file.flush()
        self._csv_file.close()
        if self._wandb is not None:
            self._wandb.finish()
