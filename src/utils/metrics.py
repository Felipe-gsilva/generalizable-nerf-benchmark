import csv
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional


class MetricsLogger:
    def __init__(
        self,
        model_name: str,
        base_dir: str = "assets/logs",
        hyperparams: Optional[Dict[str, Any]] = None,
        group_dir: Optional[str] = None,
        timestamp: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initializes the MetricsLogger.

        Args:
            model_name: Name of the model (e.g., 'Classifier', 'DCGAN').
            base_dir: Base directory for logs.
            hyperparams: Optional dictionary of hyperparameters to log.
            group_dir: Optional subdirectory under base_dir to group runs.
            timestamp: Optional timestamp to reuse across runs.
            run_id: Optional run identifier to avoid log collisions.
        """
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        group_path = Path(group_dir) if group_dir else Path(self.model_name)
        timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        # Create directory if it doesn't exist
        # Directory name now includes timestamp for uniqueness per run
        self.log_dir = self.base_dir / group_path / timestamp
        if run_id:
            self.log_dir = self.log_dir / run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "metrics.csv"
        # Save hyperparameters if provided
        if hyperparams:
            with open(self.log_dir / "config.json", "w") as f:
                json.dump(hyperparams, f, indent=4)

        self.file_handle = None
        self.writer = None
        self.headers = None

    def log(self, **metrics: Any):
        """
        Logs a dictionary of metrics to the CSV file.

        Args:
            metrics: Key-value pairs of metrics to log.
        """
        if self.file_handle is None:
            self.file_handle = open(self.log_file, mode="w", newline="")
            self.headers = list(metrics.keys())
            self.writer = csv.DictWriter(self.file_handle, fieldnames=self.headers)
            self.writer.writeheader()

        safe_metrics = {k: v for k, v in metrics.items() if k in self.headers}

        if self.writer:
            self.writer.writerow(safe_metrics)
            self.file_handle.flush()

    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
