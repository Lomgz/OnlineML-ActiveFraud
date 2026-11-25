from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.active_learning import CausalQuantileThreshold, VerificationController
from src.metrics import compute_metrics
from src.models import BatchXGBoostWrapper, OnlineFraudModel, SemiSupervisedHST
from src.simulator import DelayedLabelQueue, prepare_temporal_split


def apply_unreported_noise(
    y: np.ndarray, unreported_rate: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_obs = y.copy()
    fraud_idx = np.where(y_obs == 1)[0]
    if len(fraud_idx) > 0 and unreported_rate > 0:
        drop_mask = rng.random(len(fraud_idx)) < unreported_rate
        y_obs[fraud_idx[drop_mask]] = 0
    return y_obs


def build_feature_dicts(X: np.ndarray, feature_cols: List[str]) -> List[Dict[str, float]]:
    return [dict(zip(feature_cols, row)) for row in X.tolist()]


def event_x_to_vec(x_dict: Dict[str, float], feature_cols: List[str]) -> List[float]:
    return [float(x_dict[c]) for c in feature_cols]


def calibrate_batch_initial_threshold(
    X_init: np.ndarray,
    y_init: np.ndarray,
    feature_cols: List[str],
    alert_rate: float,
    seed: int,
) -> float:
    """Estimate causal Chunk-0 threshold using tail 15% of matured warm-up."""
    n = len(y_init)
    split = int(n * 0.85)
    if split < 100 or len(np.unique(y_init[:split])) < 2:
        return 0.5
    cal_clf = BatchXGBoostWrapper(feature_cols, mode="static", seed=seed)
    cal_clf.fit(X_init[:split], y_init[:split])
    val_scores = cal_clf.predict_proba_batch(X_init[split:])
    q = max(0.0, min(1.0, 1.0 - alert_rate))
    raw = float(np.quantile(val_scores, q))
    return min(0.95, max(1e-4, raw))


def run_online_strategy(
    strategy: str,
    base_oml: OnlineFraudModel,
    base_hst: SemiSupervisedHST,
    init_threshold: float,
    init_top_threshold: float,
    init_disc_threshold: float,
    pending_times: np.ndarray,
    pending_dicts: List[Dict[str, float]],
    pending_y: np.ndarray,
    test_times: np.ndarray,
    test_dicts: List[Dict[str, float]],
    test_y: np.ndarray,
    feature_cols: List[str],
    chunk_size: int,
    budget_per_chunk: int,
    alert_rate: float,
    confirm_window_sec: float,
    min_fraud_delay_sec: float,
    fast_track_delay_sec: float,
    unreported_fraud_rate: float,
    hybrid_top_ratio: float,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[int, Tuple[List[List[float]], List[int]]]]:
    t0_wall = time.perf_counter()
    oml = copy.deepcopy(base_oml)
    use_hst = strategy == "B4_Hybrid"
    hst = copy.deepcopy(base_hst) if use_hst else None

    queue = DelayedLabelQueue(
        confirm_window_sec=confirm_window_sec,
        min_fraud_delay_sec=min_fraud_delay_sec,
        fast_track_delay_sec=fast_track_delay_sec,
        unreported_fraud_rate=unreported_fraud_rate,
        seed=seed,
    )

    # Pre-load in-flight warm-up transactions into the delayed queue
    for idx in range(len(pending_y)):
        queue.push(
            tx_id=-(idx + 1),
            t_arrive=float(pending_times[idx]),
            x=pending_dicts[idx],
            y_true=int(pending_y[idx]),
            is_fast_track=False,
        )

    threshold_mgr = CausalQuantileThreshold(
        alert_rate=alert_rate,
        initial_threshold=init_threshold,
    )
    ctrl = VerificationController(
        strategy=strategy,
        window_size=chunk_size,
        budget_per_window=budget_per_chunk,
        hybrid_top_ratio=hybrid_top_ratio,
        initial_top_threshold=init_top_threshold,
        initial_disc_threshold=init_disc_threshold,
        seed=seed,
    )

    n_test = len(test_y)
    y_scores = np.zeros(n_test, dtype=np.float64)
    y_preds = np.zeros(n_test, dtype=np.int32)
    latencies_ms = np.zeros(n_test, dtype=np.float64)

    chunk_rows: List[Dict[str, float]] = []
    matured_by_chunk: Dict[int, Tuple[List[List[float]], List[int]]] = {}

    chunk_id = 0
    chunk_start = 0
    chunk_matured_X: List[List[float]] = []
    chunk_matured_y: List[int] = []
    chunk_queries = 0
    chunk_ft_frauds = 0

    for i in range(n_test):
        t_now = float(test_times[i])
        x_i = test_dicts[i]
        y_true_i = int(test_y[i])

        # 1. Process all labels that have matured at or before t_now
        ready_events = queue.pop_ready(t_now)
        for ev in ready_events:
            oml.learn_one(ev.x, ev.y_observed)
            if use_hst and hst is not None:
                hst.learn_one(ev.x, ev.y_observed)
            chunk_matured_X.append(event_x_to_vec(ev.x, feature_cols))
            chunk_matured_y.append(int(ev.y_observed))

        # 2. Causal prediction + Active Learning decision (timed for SLA)
        t_pred_start = time.perf_counter()
        p_hat = oml.predict_proba_one(x_i)
        hst_score = hst.score_one(x_i) if (use_hst and hst is not None) else 0.0
        theta = threshold_mgr.get_threshold()
        y_pred_i = 1 if p_hat >= theta else 0
        dec = ctrl.decide(p_hat=p_hat, hst_score=hst_score, decision_threshold=theta)
        latencies_ms[i] = (time.perf_counter() - t_pred_start) * 1000.0

        y_scores[i] = p_hat
        y_preds[i] = y_pred_i
        threshold_mgr.record_score(p_hat)

        if dec.should_query:
            chunk_queries += 1
            if y_true_i == 1:
                ctrl.record_outcome(1)
                chunk_ft_frauds += 1

        # 3. Enqueue transaction for delayed or fast-track feedback
        queue.push(
            tx_id=i,
            t_arrive=t_now,
            x=x_i,
            y_true=y_true_i,
            is_fast_track=dec.should_query,
            p_hat_at_arrive=p_hat,
        )

        # 4. Chunk boundary accounting & causal threshold update
        is_chunk_end = ((i + 1 - chunk_start) == chunk_size) or (i == n_test - 1)
        if is_chunk_end:
            c_slice = slice(chunk_start, i + 1)
            k_chunk = max(1, int(round(alert_rate * (i + 1 - chunk_start))))
            cm = compute_metrics(
                y_true=test_y[c_slice],
                y_scores=y_scores[c_slice],
                y_pred=y_preds[c_slice],
                k=k_chunk,
                latencies_ms=latencies_ms[c_slice],
                fast_track_queries=chunk_queries,
                fast_track_frauds=chunk_ft_frauds,
            )
            cm["model"] = strategy
            cm["chunk_id"] = chunk_id
            cm["chunk_n"] = i + 1 - chunk_start
            cm["chunk_frauds"] = int(np.sum(test_y[c_slice] == 1))
            chunk_rows.append(cm)

            matured_by_chunk[chunk_id] = (chunk_matured_X, chunk_matured_y)
            chunk_matured_X = []
            chunk_matured_y = []
            chunk_queries = 0
            chunk_ft_frauds = 0

            threshold_mgr.end_chunk()
            ctrl.end_window()

            chunk_id += 1
            chunk_start = i + 1

    k_global = max(1, int(round(alert_rate * n_test)))
    summary = compute_metrics(
        y_true=test_y,
        y_scores=y_scores,
        y_pred=y_preds,
        k=k_global,
        latencies_ms=latencies_ms,
        fast_track_queries=ctrl.total_queries,
        fast_track_frauds=ctrl.total_true_frauds_found,
    )
    summary["model"] = strategy
    summary["family"] = "Online_ARF"
    summary["wall_time_sec"] = round(time.perf_counter() - t0_wall, 2)
    return summary, chunk_rows, matured_by_chunk


def run_batch_strategy(
    name: str,
    mode: str,
    X_init: np.ndarray,
    y_init: np.ndarray,
    init_threshold: float,
    test_X: np.ndarray,
    test_y: np.ndarray,
    feature_cols: List[str],
    chunk_size: int,
    alert_rate: float,
    matured_by_chunk: Dict[int, Tuple[List[List[float]], List[int]]],
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    t0_wall = time.perf_counter()
    clf = BatchXGBoostWrapper(feature_cols=feature_cols, mode=mode, seed=seed)
    clf.fit(X_init, y_init)

    pool_X_blocks: List[np.ndarray] = [X_init]
    pool_y_blocks: List[np.ndarray] = [y_init]

    threshold_mgr = CausalQuantileThreshold(
        alert_rate=alert_rate,
        initial_threshold=init_threshold,
    )

    n_test = len(test_y)
    y_scores = np.zeros(n_test, dtype=np.float64)
    y_preds = np.zeros(n_test, dtype=np.int32)
    chunk_rows: List[Dict[str, float]] = []

    # Measure real single-transaction inference latency on up to 500 samples
    lat_n = min(500, n_test)
    sample_latencies = np.zeros(lat_n, dtype=np.float64)
    for j in range(lat_n):
        row_dict = dict(zip(feature_cols, test_X[j].tolist()))
        ts = time.perf_counter()
        clf.predict_proba_one(row_dict)
        sample_latencies[j] = (time.perf_counter() - ts) * 1000.0

    chunk_id = 0
    for chunk_start in range(0, n_test, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_test)
        X_chunk = test_X[chunk_start:chunk_end]
        y_chunk = test_y[chunk_start:chunk_end]

        scores_chunk = clf.predict_proba_batch(X_chunk)
        theta = threshold_mgr.get_threshold()
        preds_chunk = (scores_chunk >= theta).astype(np.int32)

        y_scores[chunk_start:chunk_end] = scores_chunk
        y_preds[chunk_start:chunk_end] = preds_chunk
        for s in scores_chunk:
            threshold_mgr.record_score(float(s))

        k_chunk = max(1, int(round(alert_rate * (chunk_end - chunk_start))))
        cm = compute_metrics(
            y_true=y_chunk,
            y_scores=scores_chunk,
            y_pred=preds_chunk,
            k=k_chunk,
            latencies_ms=sample_latencies,
            fast_track_queries=0,
            fast_track_frauds=0,
        )
        cm["model"] = name
        cm["chunk_id"] = chunk_id
        cm["chunk_n"] = chunk_end - chunk_start
        cm["chunk_frauds"] = int(np.sum(y_chunk == 1))
        chunk_rows.append(cm)

        threshold_mgr.end_chunk()

        # Retrain at chunk boundary ONLY with matured labels from the paired stream
        if mode in {"periodic", "periodic_al"}:
            new_X_list, new_y_list = matured_by_chunk.get(chunk_id, ([], []))
            if new_X_list:
                pool_X_blocks.append(np.asarray(new_X_list, dtype=np.float32))
                pool_y_blocks.append(np.asarray(new_y_list, dtype=np.int32))
                X_pool = np.vstack(pool_X_blocks)
                y_pool = np.concatenate(pool_y_blocks)
                clf.update_if_allowed(X_pool, y_pool)

        chunk_id += 1

    k_global = max(1, int(round(alert_rate * n_test)))
    summary = compute_metrics(
        y_true=test_y,
        y_scores=y_scores,
        y_pred=y_preds,
        k=k_global,
        latencies_ms=sample_latencies,
        fast_track_queries=0,
        fast_track_frauds=0,
    )
    summary["model"] = name
    summary["family"] = "Batch_XGBoost"
    summary["wall_time_sec"] = round(time.perf_counter() - t0_wall, 2)
    return summary, chunk_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Active Learning Selective Verification Benchmark")
    parser.add_argument("--data", type=str, default="data/creditcard.csv")
    parser.add_argument("--warmup-ratio", type=float, default=0.5)
    parser.add_argument("--max-warmup-rows", type=int, default=0, help="0 = use all matured warm-up rows")
    parser.add_argument("--max-test-rows", type=int, default=0, help="0 = use full suffix")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--beta", type=float, default=0.005, help="Verification budget & alert rate")
    parser.add_argument("--hybrid-top-ratio", type=float, default=0.70)
    parser.add_argument("--confirm-window-sec", type=float, default=21600.0)
    parser.add_argument("--min-fraud-delay-sec", type=float, default=1800.0)
    parser.add_argument("--fast-track-delay-sec", type=float, default=300.0)
    parser.add_argument("--unreported-rate", type=float, default=0.05)
    parser.add_argument("--n-models", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading {args.data} ...")
    df = pd.read_csv(args.data)
    warmup_matured, warmup_pending, test_df, feature_cols = prepare_temporal_split(
        df,
        warmup_ratio=args.warmup_ratio,
        confirm_window_sec=args.confirm_window_sec,
    )

    if args.max_warmup_rows > 0 and len(warmup_matured) > args.max_warmup_rows:
        warmup_matured = warmup_matured.iloc[-args.max_warmup_rows :].reset_index(drop=True)
    if args.max_test_rows > 0 and len(test_df) > args.max_test_rows:
        test_df = test_df.iloc[: args.max_test_rows].reset_index(drop=True)

    budget_per_chunk = max(1, int(round(args.beta * args.chunk_size)))
    print(
        f"Split ready | warmup_matured={len(warmup_matured)} (frauds={int(warmup_matured['Class'].sum())}), "
        f"warmup_pending={len(warmup_pending)}, test={len(test_df)} (frauds={int(test_df['Class'].sum())}), "
        f"chunk_size={args.chunk_size}, budget_per_chunk={budget_per_chunk} (beta={args.beta:.4f})"
    )

    # Prepare numpy arrays & feature dicts once
    X_warm = warmup_matured[feature_cols].to_numpy(dtype=np.float32)
    y_warm_true = warmup_matured["Class"].to_numpy(dtype=np.int32)
    y_warm_obs = apply_unreported_noise(y_warm_true, args.unreported_rate, seed=args.seed)
    warm_dicts = build_feature_dicts(X_warm, feature_cols)

    pending_times = warmup_pending["Time"].to_numpy(dtype=np.float64)
    pending_X = warmup_pending[feature_cols].to_numpy(dtype=np.float32)
    pending_dicts = build_feature_dicts(pending_X, feature_cols)
    pending_y = warmup_pending["Class"].to_numpy(dtype=np.int32)

    test_times = test_df["Time"].to_numpy(dtype=np.float64)
    test_X = test_df[feature_cols].to_numpy(dtype=np.float32)
    test_dicts = build_feature_dicts(test_X, feature_cols)
    test_y = test_df["Class"].to_numpy(dtype=np.int32)

    # Shared warm-up for Online ARF + Semi-Supervised HST (ensures identical starting state)
    print(f"[2/4] Warming up shared ARF (n_models={args.n_models}) + Semi-Supervised HST ...")
    t_warm0 = time.perf_counter()
    base_oml = OnlineFraudModel(n_models=args.n_models, seed=args.seed)
    base_hst = SemiSupervisedHST(seed=args.seed)

    tail_start = max(0, len(y_warm_obs) - args.chunk_size)
    warm_tail_p: List[float] = []
    warm_tail_disc: List[float] = []

    for i in range(len(y_warm_obs)):
        x_i = warm_dicts[i]
        y_i = int(y_warm_obs[i])
        if i >= tail_start:
            p_i = base_oml.predict_proba_one(x_i)
            s_i = base_hst.score_one(x_i)
            warm_tail_p.append(p_i)
            warm_tail_disc.append(s_i - p_i)
        base_oml.learn_one(x_i, y_i)
        base_hst.learn_one(x_i, y_i)

    p_arr = np.asarray(warm_tail_p, dtype=np.float64)
    d_arr = np.asarray(warm_tail_disc, dtype=np.float64)
    init_oml_thresh = min(0.95, max(1e-4, float(np.quantile(p_arr, 1.0 - args.beta))))
    top_rate = max(1e-4, (budget_per_chunk * args.hybrid_top_ratio) / max(1, len(p_arr)))
    disc_rate = max(1e-4, (budget_per_chunk * (1.0 - args.hybrid_top_ratio)) / max(1, len(d_arr)))
    init_top_thresh = float(np.quantile(p_arr, max(0.0, 1.0 - top_rate)))
    init_disc_thresh = float(np.quantile(d_arr, max(0.0, 1.0 - disc_rate)))
    print(
        f"Warm-up done in {time.perf_counter() - t_warm0:.1f}s | "
        f"init_thresh={init_oml_thresh:.4f}, init_top={init_top_thresh:.4f}, init_disc={init_disc_thresh:.4f}"
    )

    # Run 5 Online strategies
    print("[3/4] Running 5 Online ARF strategies (B0-B4) ...")
    summary_rows: List[Dict[str, float]] = []
    all_chunk_rows: List[Dict[str, float]] = []
    matured_standard: Dict[int, Tuple[List[List[float]], List[int]]] = {}
    matured_b4_al: Dict[int, Tuple[List[List[float]], List[int]]] = {}

    for strat in ["B0_NoAL", "B1_Random", "B2_TopRisk", "B3_Margin", "B4_Hybrid"]:
        print(f"  -> Running {strat} ...")
        s_row, c_rows, matured_map = run_online_strategy(
            strategy=strat,
            base_oml=base_oml,
            base_hst=base_hst,
            init_threshold=init_oml_thresh,
            init_top_threshold=init_top_thresh,
            init_disc_threshold=init_disc_thresh,
            pending_times=pending_times,
            pending_dicts=pending_dicts,
            pending_y=pending_y,
            test_times=test_times,
            test_dicts=test_dicts,
            test_y=test_y,
            feature_cols=feature_cols,
            chunk_size=args.chunk_size,
            budget_per_chunk=budget_per_chunk,
            alert_rate=args.beta,
            confirm_window_sec=args.confirm_window_sec,
            min_fraud_delay_sec=args.min_fraud_delay_sec,
            fast_track_delay_sec=args.fast_track_delay_sec,
            unreported_fraud_rate=args.unreported_rate,
            hybrid_top_ratio=args.hybrid_top_ratio,
            seed=args.seed,
        )
        summary_rows.append(s_row)
        all_chunk_rows.extend(c_rows)
        if strat == "B0_NoAL":
            matured_standard = matured_map
        elif strat == "B4_Hybrid":
            matured_b4_al = matured_map
        print(
            f"     {strat}: AP={s_row['AP']:.4f}, Pr@k={s_row['Pr@k']:.4f}, "
            f"Recall={s_row['Recall']:.4f}, FP={int(s_row['FP'])}, "
            f"Queries={int(s_row['FastTrack_Queries'])}, Yield={s_row['Verification_Yield']:.4f}, "
            f"lat={s_row['mean_latency_ms']:.3f}ms ({s_row['wall_time_sec']}s)"
        )

    # Run 3 Batch XGBoost baselines with strict Information Parity
    print("[4/4] Running 3 Batch XGBoost baselines ...")
    init_batch_thresh = calibrate_batch_initial_threshold(
        X_warm, y_warm_obs, feature_cols, alert_rate=args.beta, seed=args.seed
    )
    batch_specs = [
        ("M_Static_Batch", "static", {}),
        ("M_Periodic_Batch", "periodic", matured_standard),
        ("M_Periodic_Batch_AL", "periodic_al", matured_b4_al),
    ]
    for b_name, b_mode, b_matured in batch_specs:
        print(f"  -> Running {b_name} ...")
        s_row, c_rows = run_batch_strategy(
            name=b_name,
            mode=b_mode,
            X_init=X_warm,
            y_init=y_warm_obs,
            init_threshold=init_batch_thresh,
            test_X=test_X,
            test_y=test_y,
            feature_cols=feature_cols,
            chunk_size=args.chunk_size,
            alert_rate=args.beta,
            matured_by_chunk=b_matured,
            seed=args.seed,
        )
        summary_rows.append(s_row)
        all_chunk_rows.extend(c_rows)
        print(
            f"     {b_name}: AP={s_row['AP']:.4f}, Pr@k={s_row['Pr@k']:.4f}, "
            f"Recall={s_row['Recall']:.4f}, FP={int(s_row['FP'])}, "
            f"lat={s_row['mean_latency_ms']:.3f}ms ({s_row['wall_time_sec']}s)"
        )

    col_order = [
        "model",
        "family",
        "AP",
        "Pr@k",
        "Recall",
        "Precision",
        "TP",
        "FP",
        "FastTrack_Queries",
        "FastTrack_Frauds",
        "Verification_Yield",
        "mean_latency_ms",
        "p99_latency_ms",
        "wall_time_sec",
    ]
    summary_df = pd.DataFrame(summary_rows)[col_order]
    chunk_df = pd.DataFrame(all_chunk_rows)

    summary_path = out_dir / "ablation_al_summary.csv"
    chunk_path = out_dir / "chunk_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    chunk_df.to_csv(chunk_path, index=False)

    print("\n=== SUMMARY TABLE ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path} and {chunk_path}")


if __name__ == "__main__":
    main()