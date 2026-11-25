from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelEvent:
    t_label: float
    seq_id: int
    tx_id: int
    x: Dict[str, float]
    y_observed: int
    y_true: int
    is_fast_track: bool
    p_hat_at_arrive: float


class DelayedLabelQueue:
    """
    Event-driven min-heap simulating realistic delayed labels + Fast-Track verification.

    - Standard channel:
        * Legitimate (y_true=0): matures at t_arrive + confirm_window_sec with y_observed=0.
        * Fraud (y_true=1):
            - With prob `unreported_fraud_rate`: never reported -> matures at
              t_arrive + confirm_window_sec with y_observed=0 (label noise).
            - Otherwise: chargeback arrives in [min_fraud_delay_sec, confirm_window_sec]
              with y_observed=1.
    - Fast-Track channel (Active Learning triggered):
        * Manual review reveals true label y_true after `fast_track_delay_sec`.
    """

    def __init__(
        self,
        confirm_window_sec: float = 21600.0,
        min_fraud_delay_sec: float = 1800.0,
        fast_track_delay_sec: float = 300.0,
        unreported_fraud_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        if fast_track_delay_sec < 0 or confirm_window_sec <= 0:
            raise ValueError("Delay parameters must be positive.")
        self.confirm_window_sec = float(confirm_window_sec)
        self.min_fraud_delay_sec = float(min(min_fraud_delay_sec, confirm_window_sec))
        self.fast_track_delay_sec = float(fast_track_delay_sec)
        self.unreported_fraud_rate = float(unreported_fraud_rate)
        self._rng = np.random.default_rng(seed)
        self._heap: List[Tuple[float, int, LabelEvent]] = []
        self._seq = 0

    def push(
        self,
        tx_id: int,
        t_arrive: float,
        x: Dict[str, float],
        y_true: int,
        is_fast_track: bool = False,
        p_hat_at_arrive: float = 0.0,
    ) -> float:
        y_true_int = int(y_true)
        if is_fast_track:
            t_label = float(t_arrive) + self.fast_track_delay_sec
            y_observed = y_true_int
        else:
            if y_true_int == 1:
                if self._rng.random() < self.unreported_fraud_rate:
                    t_label = float(t_arrive) + self.confirm_window_sec
                    y_observed = 0
                else:
                    delay = float(
                        self._rng.uniform(self.min_fraud_delay_sec, self.confirm_window_sec)
                    )
                    t_label = float(t_arrive) + delay
                    y_observed = 1
            else:
                t_label = float(t_arrive) + self.confirm_window_sec
                y_observed = 0

        self._seq += 1
        event = LabelEvent(
            t_label=t_label,
            seq_id=self._seq,
            tx_id=int(tx_id),
            x=x,
            y_observed=y_observed,
            y_true=y_true_int,
            is_fast_track=bool(is_fast_track),
            p_hat_at_arrive=float(p_hat_at_arrive),
        )
        heapq.heappush(self._heap, (t_label, self._seq, event))
        return t_label

    def pop_ready(self, t_current: float) -> List[LabelEvent]:
        ready: List[LabelEvent] = []
        t_now = float(t_current)
        while self._heap and self._heap[0][0] <= t_now:
            _, _, event = heapq.heappop(self._heap)
            ready.append(event)
        return ready

    def pending_count(self) -> int:
        return len(self._heap)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return strictly model features (never include Time or Class)."""
    excluded = {"Time", "Class"}
    return [c for c in df.columns if c not in excluded]


def row_to_feature_dict(row: Any, feature_cols: List[str]) -> Dict[str, float]:
    return {col: float(row[col]) for col in feature_cols}


def prepare_temporal_split(
    df: pd.DataFrame,
    warmup_ratio: float = 0.5,
    confirm_window_sec: float = 21600.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Temporal split preserving strict Information Parity at t_split:
    - warmup_matured: prefix rows whose label is guaranteed matured at t_split
      (t_arrive + confirm_window_sec <= t_split).
    - warmup_pending: prefix rows still in-flight at t_split (must enter DelayedLabelQueue).
    - test_df: suffix rows streamed sequentially for evaluation.
    """
    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    split_idx = int(len(ordered) * warmup_ratio)
    if split_idx <= 0 or split_idx >= len(ordered):
        raise ValueError("Invalid warmup_ratio for dataset size.")

    t_split = float(ordered.iloc[split_idx]["Time"])
    prefix = ordered.iloc[:split_idx]
    test_df = ordered.iloc[split_idx:].reset_index(drop=True)

    matured_mask = (prefix["Time"].astype(float) + float(confirm_window_sec)) <= t_split
    warmup_matured = prefix.loc[matured_mask].reset_index(drop=True)
    warmup_pending = prefix.loc[~matured_mask].reset_index(drop=True)
    feature_cols = get_feature_columns(ordered)

    return warmup_matured, warmup_pending, test_df, feature_cols