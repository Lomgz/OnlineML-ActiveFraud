from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    n = len(y_true)
    if n == 0 or k <= 0:
        return 0.0
    k_eff = min(int(k), n)
    top_idx = np.argsort(y_scores, kind="mergesort")[::-1][:k_eff]
    return float(np.sum(y_true[top_idx] == 1) / k_eff)


def compute_metrics(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    y_pred: Sequence[int],
    k: int,
    latencies_ms: Sequence[float],
    fast_track_queries: int = 0,
    fast_track_frauds: int = 0,
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.int32)
    ys = np.asarray(y_scores, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.int32)
    lat = np.asarray(latencies_ms, dtype=np.float64)

    n_pos = int(np.sum(yt == 1))
    ap = float(average_precision_score(yt, ys)) if n_pos > 0 else 0.0
    pr_k = precision_at_k(yt, ys, k)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    verif_yield = (
        float(fast_track_frauds / fast_track_queries) if fast_track_queries > 0 else 0.0
    )

    mean_lat = float(np.mean(lat)) if len(lat) > 0 else 0.0
    p99_lat = float(np.percentile(lat, 99)) if len(lat) > 0 else 0.0

    return {
        "AP": ap,
        "Pr@k": pr_k,
        "Recall": recall,
        "Precision": precision,
        "TP": float(tp),
        "FP": float(fp),
        "FastTrack_Queries": float(fast_track_queries),
        "FastTrack_Frauds": float(fast_track_frauds),
        "Verification_Yield": verif_yield,
        "mean_latency_ms": mean_lat,
        "p99_latency_ms": p99_lat,
    }