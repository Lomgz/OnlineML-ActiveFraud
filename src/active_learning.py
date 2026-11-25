from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


class CausalQuantileThreshold:
    """
    Computes decision threshold strictly from COMPLETED historical chunks/windows.
    Scores from the current chunk never affect predictions inside the current chunk.
    """

    def __init__(
        self,
        alert_rate: float = 0.005,
        initial_threshold: float = 0.5,
        min_threshold: float = 1e-4,
        max_threshold: float = 0.95,
        max_history: int = 50000,
    ) -> None:
        self.alert_rate = float(alert_rate)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self._completed_scores: Deque[float] = deque(maxlen=max_history)
        self._current_chunk_scores: List[float] = []
        self._threshold = float(initial_threshold)

    def get_threshold(self) -> float:
        return self._threshold

    def record_score(self, p_hat: float) -> None:
        self._current_chunk_scores.append(float(p_hat))

    def end_chunk(self) -> float:
        if self._current_chunk_scores:
            self._completed_scores.extend(self._current_chunk_scores)
            self._current_chunk_scores.clear()
        if self._completed_scores:
            q = max(0.0, min(1.0, 1.0 - self.alert_rate))
            raw = float(np.quantile(np.asarray(self._completed_scores, dtype=np.float64), q))
            self._threshold = min(self.max_threshold, max(self.min_threshold, raw))
        return self._threshold

    def recalibrate_on_drift(self, recent_tail: int = 2000) -> float:
        """
        Recalibrate threshold strictly from the most recent completed historical scores
        upon an ADWIN drift event (never touching current-chunk scores).
        """
        if self._completed_scores:
            tail = list(self._completed_scores)[-max(100, int(recent_tail)) :]
            q = max(0.0, min(1.0, 1.0 - self.alert_rate))
            raw = float(np.quantile(np.asarray(tail, dtype=np.float64), q))
            self._threshold = min(self.max_threshold, max(self.min_threshold, raw))
        else:
            self._threshold = max(self.min_threshold, self._threshold * 0.90)
        return self._threshold


@dataclass
class QueryDecision:
    should_query: bool
    reason: str


class VerificationController:
    """
    Stream-based Active Learning controller with a HARD per-window budget cap B.
    Strategies:
      - B0_NoAL     : never queries (B = 0).
      - B1_Random   : uniform random selection paced across window within cap B.
      - B2_TopRisk  : exploits highest p_hat via causal quantile threshold within cap B.
      - B3_Margin   : queries uncertain points |p_hat - theta| <= margin (adaptive margin).
      - B4_Hybrid   : splits cap B into Top-Risk (default 70%) + HST-Tree Discordance (30%).
    """

    VALID_STRATEGIES = {"B0_NoAL", "B1_Random", "B2_TopRisk", "B3_Margin", "B4_Hybrid"}

    def __init__(
        self,
        strategy: str,
        window_size: int = 5000,
        budget_per_window: int = 25,
        hybrid_top_ratio: float = 0.70,
        initial_top_threshold: float = 0.30,
        initial_disc_threshold: float = 0.40,
        initial_margin: float = 0.15,
        seed: int = 42,
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy '{strategy}'.")
        self.strategy = strategy
        self.window_size = max(1, int(window_size))
        self.budget_per_window = 0 if strategy == "B0_NoAL" else max(0, int(budget_per_window))
        self.hybrid_top_ratio = float(hybrid_top_ratio)

        self.budget_top = int(round(self.budget_per_window * self.hybrid_top_ratio))
        self.budget_disc = max(0, self.budget_per_window - self.budget_top)

        self.top_threshold = float(initial_top_threshold)
        self.disc_threshold = float(initial_disc_threshold)
        self.margin_width = float(initial_margin)

        self._rng = np.random.default_rng(seed)

        # Per-window counters
        self.seen_in_window = 0
        self.used_in_window = 0
        self.used_top_in_window = 0
        self.used_disc_in_window = 0
        self._window_p_hats: List[float] = []
        self._window_discs: List[float] = []

        # Cumulative statistics
        self.total_queries = 0
        self.total_true_frauds_found = 0
        self.total_top_queries = 0
        self.total_disc_queries = 0

    def decide(
        self,
        p_hat: float,
        hst_score: float,
        decision_threshold: float,
    ) -> QueryDecision:
        p = float(p_hat)
        s_hst = float(hst_score)
        delta_disc = s_hst - p

        self.seen_in_window += 1
        self._window_p_hats.append(p)
        self._window_discs.append(delta_disc)

        if self.strategy == "B0_NoAL" or self.used_in_window >= self.budget_per_window:
            return QueryDecision(False, "budget_exhausted_or_no_al")

        decision = QueryDecision(False, "skip")

        if self.strategy == "B1_Random":
            remaining_tx = max(1, self.window_size - self.seen_in_window + 1)
            remaining_budget = self.budget_per_window - self.used_in_window
            prob = min(1.0, remaining_budget / remaining_tx)
            if self._rng.random() < prob:
                decision = QueryDecision(True, "random")

        elif self.strategy == "B2_TopRisk":
            if p >= self.top_threshold:
                decision = QueryDecision(True, "top_risk")
                self.top_threshold = min(0.99, self.top_threshold * 1.02 + 1e-4)
            else:
                self.top_threshold = max(1e-6, self.top_threshold * 0.9995)

        elif self.strategy == "B3_Margin":
            dist = abs(p - float(decision_threshold))
            if dist <= self.margin_width:
                decision = QueryDecision(True, "margin")
                self.margin_width = max(0.005, self.margin_width * 0.95)
            else:
                self.margin_width = min(0.45, self.margin_width * 1.001)

        elif self.strategy == "B4_Hybrid":
            if self.used_top_in_window < self.budget_top and p >= self.top_threshold:
                decision = QueryDecision(True, "top_risk")
                self.used_top_in_window += 1
                self.total_top_queries += 1
                self.top_threshold = min(0.99, self.top_threshold * 1.02 + 1e-4)
            elif self.used_disc_in_window < self.budget_disc and delta_disc >= self.disc_threshold:
                decision = QueryDecision(True, "discordance")
                self.used_disc_in_window += 1
                self.total_disc_queries += 1
                self.disc_threshold = min(0.99, self.disc_threshold * 1.02 + 1e-4)
            else:
                self.top_threshold = max(1e-6, self.top_threshold * 0.9995)
                self.disc_threshold = max(1e-4, self.disc_threshold * 0.9995)

        if decision.should_query:
            self.used_in_window += 1
            self.total_queries += 1

        return decision

    def record_outcome(self, y_true: int) -> None:
        if int(y_true) == 1:
            self.total_true_frauds_found += 1

    def end_window(self) -> Tuple[float, float]:
        """Reset per-window budget and recalibrate causal thresholds from finished window."""
        if self._window_p_hats and self.budget_per_window > 0:
            p_arr = np.asarray(self._window_p_hats, dtype=np.float64)
            d_arr = np.asarray(self._window_discs, dtype=np.float64)

            if self.strategy == "B4_Hybrid":
                top_rate = max(1e-4, self.budget_top / max(1, len(p_arr)))
                disc_rate = max(1e-4, self.budget_disc / max(1, len(d_arr)))
                self.top_threshold = float(np.quantile(p_arr, max(0.0, 1.0 - top_rate)))
                self.disc_threshold = float(np.quantile(d_arr, max(0.0, 1.0 - disc_rate)))
            else:
                rate = max(1e-4, self.budget_per_window / max(1, len(p_arr)))
                self.top_threshold = float(np.quantile(p_arr, max(0.0, 1.0 - rate)))

        self.seen_in_window = 0
        self.used_in_window = 0
        self.used_top_in_window = 0
        self.used_disc_in_window = 0
        self._window_p_hats.clear()
        self._window_discs.clear()
        return self.top_threshold, self.disc_threshold

    def set_discordance_ratio(self, disc_ratio: float) -> None:
        """Dynamically adjust B4_Hybrid Top-Risk/Discordance split within hard cap B."""
        clamped_disc = min(0.90, max(0.0, float(disc_ratio)))
        self.hybrid_top_ratio = 1.0 - clamped_disc
        self.budget_top = int(round(self.budget_per_window * self.hybrid_top_ratio))
        self.budget_disc = max(0, self.budget_per_window - self.budget_top)

    @property
    def verification_yield(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_true_frauds_found / self.total_queries