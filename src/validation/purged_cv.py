from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardConfig:
    """
    Purged Walk-Forward settings.

    train_size: number of bars in the training window
    test_size: number of bars in the test window
    step_size: number of bars to advance each fold
    purge_size: number of bars to purge from END of train to avoid label overlap
               (usually equal to label horizon in bars)
    embargo_size: optional buffer bars (kept for future extensions)
    """
    train_size: int
    test_size: int
    step_size: int
    purge_size: int
    embargo_size: int = 0


class PurgedWalkForwardSplitter:
    """
    Walk-forward splitter with purge.

    For each fold:
      - Train window: [test_start - train_size, test_start)
      - Then purge the last `purge_size` bars of the train window
      - Test window:  [test_start, test_start + test_size)
    """

    def __init__(self, cfg: PurgedWalkForwardConfig):
        if cfg.train_size <= 0 or cfg.test_size <= 0 or cfg.step_size <= 0:
            raise ValueError("train_size, test_size, step_size must be positive")
        if cfg.purge_size < 0 or cfg.embargo_size < 0:
            raise ValueError("purge_size and embargo_size must be >= 0")
        if cfg.train_size <= cfg.purge_size:
            raise ValueError("train_size must be > purge_size")
        self.cfg = cfg

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        cfg = self.cfg
        start = cfg.train_size

        while True:
            test_start = start
            test_end = test_start + cfg.test_size
            if test_end > n_samples:
                break

            train_start = test_start - cfg.train_size
            train_end = test_start

            purged_train_end = train_end - cfg.purge_size
            if purged_train_end <= train_start:
                start += cfg.step_size
                continue

            train_idx = np.arange(train_start, purged_train_end, dtype=np.int64)
            test_idx = np.arange(test_start, test_end, dtype=np.int64)

            yield train_idx, test_idx

            start += cfg.step_size


def assert_time_sorted(df: pd.DataFrame, time_col: Optional[str] = None) -> None:
    """
    Ensure dataframe is sorted ascending by time (index or time_col).
    """
    if time_col is None:
        t = df.index
    else:
        t = df[time_col]

    if isinstance(t, pd.DatetimeIndex):
        if not t.is_monotonic_increasing:
            raise ValueError("DataFrame datetime index must be sorted ascending.")
    else:
        t = pd.Series(t)
        if not t.is_monotonic_increasing:
            raise ValueError("DataFrame time column must be sorted ascending.")
