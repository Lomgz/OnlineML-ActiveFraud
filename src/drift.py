from __future__ import annotations

from typing import Any, Dict, List

from river.drift import ADWIN


class ADWINDriftManager:
    """
    Explicit river.drift.ADWIN monitor driven by the 0-1 prediction error stream
    (int(y_pred_at_arrive != y_observed)) strictly at label arrival time (t_label <= t_now).

    On drift detection:
      1. Logs the drift event (timestamp, matured step, estimated error rate, action).
      2. Activates a cooldown window that boosts B4_Hybrid discordance exploration ratio
         (default 0.30 -> 0.50) and signals threshold recalibration.
    """

    def __init__(
        self,
        delta: float = 0.002,
        cooldown_duration: int = 500,
        base_disc_ratio: float = 0.30,
        boosted_disc_ratio: float = 0.50,
    ) -> None:
        self.adwin = ADWIN(delta=delta)
        self.cooldown_duration = max(1, int(cooldown_duration))
        self.base_disc_ratio = float(base_disc_ratio)
        self.boosted_disc_ratio = float(boosted_disc_ratio)

        self.cooldown_remaining = 0
        self.total_processed = 0
        self.drift_events: List[Dict[str, Any]] = []

    def update(self, y_pred: int, y_true: int, current_time: float) -> bool:
        """Feed 0-1 prediction error into ADWIN at label maturity time."""
        self.total_processed += 1
        error = int(int(y_pred) != int(y_true))
        self.adwin.update(error)

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if bool(getattr(self.adwin, "drift_detected", False)):
            self.cooldown_remaining = self.cooldown_duration
            self.drift_events.append(
                {
                    "timestamp": float(current_time),
                    "step": self.total_processed,
                    "error_rate": float(getattr(self.adwin, "estimation", float(error))),
                    "action": (
                        f"Boost Discordance ({int(self.boosted_disc_ratio * 100)}%) "
                        "& Recalibrate Threshold"
                    ),
                }
            )
            return True
        return False

    def get_current_disc_ratio(self) -> float:
        """Return boosted discordance ratio during cooldown, else base ratio."""
        if self.cooldown_remaining > 0:
            return self.boosted_disc_ratio
        return self.base_disc_ratio

    def is_in_cooldown(self) -> bool:
        return self.cooldown_remaining > 0
