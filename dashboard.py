from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.active_learning import CausalQuantileThreshold, VerificationController
from src.metrics import compute_metrics
from src.models import OnlineFraudModel, SemiSupervisedHST
from src.simulator import DelayedLabelQueue, prepare_temporal_split

RESULTS_DIR = ROOT_DIR / "results"
DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"


st.set_page_config(
    page_title="OnlineML-ActiveFraud Dashboard",
    layout="wide",
)


@st.cache_data
def load_results():
    summary_path = RESULTS_DIR / "ablation_al_summary.csv"
    chunk_path = RESULTS_DIR / "chunk_metrics.csv"
    if not summary_path.exists() or not chunk_path.exists():
        return None, None
    return pd.read_csv(summary_path), pd.read_csv(chunk_path)


@st.cache_data
def load_split_for_replay(confirm_window_sec: float = 21600.0):
    df = pd.read_csv(DATA_PATH)
    return prepare_temporal_split(
        df,
        warmup_ratio=0.5,
        confirm_window_sec=confirm_window_sec,
    )


def render_comparison_tab(summary: pd.DataFrame) -> None:
    b0 = summary.loc[summary["model"] == "B0_NoAL"].iloc[0]
    b4 = summary.loc[summary["model"] == "B4_Hybrid"].iloc[0]
    m_static = summary.loc[summary["model"] == "M_Static_Batch"].iloc[0]

    ap_gain_pct = ((b4["AP"] - b0["AP"]) / max(1e-9, b0["AP"])) * 100.0
    gap_closed_pct = (
        (b4["AP"] - b0["AP"]) / max(1e-9, m_static["AP"] - b0["AP"])
    ) * 100.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("B4_Hybrid AP (PR-AUC)", f"{b4['AP']:.4f}", f"+{ap_gain_pct:.1f}% vs B0_NoAL")
    c2.metric(
        "Verification Yield (B4)",
        f"{b4['Verification_Yield'] * 100:.2f}%",
        f"{int(b4['FastTrack_Frauds'])}/{int(b4['FastTrack_Queries'])} gian lận thật",
    )
    c3.metric("Thu hẹp khoảng cách Batch", f"{gap_closed_pct:.1f}%", "AP gap vs Static XGB")
    c4.metric(
        "Độ trễ suy luận (B4)",
        f"{b4['mean_latency_ms']:.3f} ms/tx",
        f"p99: {b4['p99_latency_ms']:.3f} ms (< 5ms SLA)",
    )

    st.divider()
    all_models = summary["model"].tolist()
    selected_models = st.multiselect(
        "Chọn các mô hình cần so sánh:",
        options=all_models,
        default=all_models,
    )
    filtered = summary[summary["model"].isin(selected_models)].copy()
    if filtered.empty:
        st.warning("Chọn ít nhất 1 mô hình để hiển thị.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("AP (PR-AUC) & Pr@k")
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        x = np.arange(len(filtered))
        w = 0.36
        ax.bar(x - w / 2, filtered["AP"], w, label="AP (PR-AUC)", color="#2563eb")
        ax.bar(x + w / 2, filtered["Pr@k"], w, label="Pr@k", color="#0ea5e9")
        ax.set_xticks(x)
        ax.set_xticklabels(filtered["model"], rotation=24, ha="right", fontsize=8.5)
        ax.set_ylim(0, min(1.05, max(filtered["AP"].max(), filtered["Pr@k"].max()) * 1.18 + 0.05))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Verification Yield & Độ trễ SLA")
        chart_mode = st.radio(
            "Chế độ biểu đồ:",
            ["Verification Yield (Nhóm Active Learning)", "Độ trễ suy luận vs SLA (5 ms)"],
            horizontal=True,
        )
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        if chart_mode.startswith("Verification"):
            al_sub = filtered[filtered["FastTrack_Queries"] > 0]
            if al_sub.empty:
                ax.text(0.5, 0.5, "Không có mô hình Active Learning được chọn", ha="center", va="center")
            else:
                x = np.arange(len(al_sub))
                bars = ax.bar(x, al_sub["Verification_Yield"] * 100.0, width=0.5, color="#10b981")
                ax.set_xticks(x)
                ax.set_xticklabels(al_sub["model"], fontsize=9)
                ax.set_ylabel("Verification Yield (%)")
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                for b, f_cnt, q_cnt in zip(bars, al_sub["FastTrack_Frauds"], al_sub["FastTrack_Queries"]):
                    h = b.get_height()
                    ax.annotate(
                        f"{h:.1f}% ({int(f_cnt)}/{int(q_cnt)})",
                        xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )
                ax.set_ylim(0, max(10.0, al_sub["Verification_Yield"].max() * 125.0 + 3.0))
        else:
            x = np.arange(len(filtered))
            w = 0.36
            ax.bar(x - w / 2, filtered["mean_latency_ms"], w, label="Mean (ms)", color="#0284c7")
            ax.bar(x + w / 2, filtered["p99_latency_ms"], w, label="p99 (ms)", color="#f97316")
            ax.axhline(5.0, color="#dc2626", linestyle="--", linewidth=1.2, label="SLA 5 ms")
            ax.set_xticks(x)
            ax.set_xticklabels(filtered["model"], rotation=24, ha="right", fontsize=8.5)
            ax.set_ylabel("ms / tx")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Bảng số liệu chi tiết (142.404 giao dịch kiểm thử, 223 gian lận)")
    display_df = filtered.copy()
    display_df["Verification_Yield"] = (display_df["Verification_Yield"] * 100.0).map(lambda v: f"{v:.2f}%")
    st.dataframe(display_df, hide_index=True)

    with st.expander("Xem 4 biểu đồ PNG xuất chính thức cho báo cáo"):
        p1, p2 = st.columns(2)
        for col, fname in [
            (p1, "fig1_overall_ap_prk.png"),
            (p2, "fig2_verification_yield.png"),
            (p1, "fig3_chunk_ap_trajectory.png"),
            (p2, "fig4_latency_sla.png"),
        ]:
            img_path = RESULTS_DIR / fname
            if img_path.exists():
                col.image(str(img_path), caption=fname)


def render_chunk_and_arch_tab(chunks: pd.DataFrame) -> None:
    st.subheader("Quỹ đạo hiệu năng theo từng Chunk thời gian (29 chunks x 5.000 tx)")
    c1, c2 = st.columns([2, 1])
    models_avail = chunks["model"].unique().tolist()
    default_models = [
        m for m in ["B0_NoAL", "B2_TopRisk", "B4_Hybrid", "M_Static_Batch", "M_Periodic_Batch_AL"]
        if m in models_avail
    ]
    chosen = c1.multiselect("Chọn mô hình vẽ quỹ đạo:", models_avail, default=default_models)
    metric_col = c2.selectbox(
        "Chỉ số theo dõi:",
        ["AP", "Pr@k", "Recall", "FP", "Verification_Yield"],
        index=0,
    )

    if chosen:
        fig, ax = plt.subplots(figsize=(10.5, 4.2))
        for m in chosen:
            sub = chunks[chunks["model"] == m].sort_values("chunk_id")
            ax.plot(sub["chunk_id"], sub[metric_col], marker="o", markersize=4, linewidth=1.7, label=m)
        ax.set_xlabel("Chunk ID (theo thứ tự thời gian)")
        ax.set_ylabel(metric_col)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", fontsize=8.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Sơ đồ kiến trúc luồng Active Learning & Selective Verification")
    st.code(
        """
[Giao dịch x_t đến tại t_arrive]
        │
        ├──► 1. Pop nhãn đã đáo hạn (t_label <= t_arrive) từ DelayedLabelQueue
        │         ├──► ARFClassifier.learn_one(x, y_obs)
        │         └──► SemiSupervisedHST.learn_one(x, y_obs)  [CHỈ học khi y_obs == 0]
        │
        ├──► 2. Suy luận nhân quả (Causal Predict)
        │         ├──► p_hat = ARF.predict_proba_one(x_t)
        │         ├──► S_HST = HST.score_one(x_t)
        │         └──► Delta_disc = S_HST - p_hat  (HST–Tree Discordance)
        │
        ├──► 3. VerificationController (Khóa trần ngân sách B = 0.5% / chunk)
        │         ├──► 70% Top-Risk (khai thác p_hat cao)
        │         └──► 30% Discordance (mở điểm mù khi HST cao nhưng ARF tự tin bỏ sót)
        │
        └──► 4. Đẩy vào DelayedLabelQueue
                  ├──► Fast-Track (nếu kích hoạt AL): trả nhãn thực sau 300s
                  └──► Standard Channel (còn lại): trả nhãn trễ 1.800s – 21.600s (+5% unreported)
        """.strip(),
        language="text",
    )


def render_live_replay_tab() -> None:
    st.subheader("Mô phỏng luồng giao dịch trực tiếp (Live Stream Replay)")
    if not DATA_PATH.exists():
        st.warning(
            "Không tìm thấy `data/creditcard.csv` trong thư mục local. "
            "Tab 1 và Tab 2 vẫn xem bình thường từ `results/*.csv`; "
            "để chạy mô phỏng live, hãy đặt `creditcard.csv` vào thư mục `data/`."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    strategy = c1.selectbox(
        "Chiến lược Active Learning:",
        ["B4_Hybrid", "B2_TopRisk", "B3_Margin", "B1_Random", "B0_NoAL"],
        index=0,
    )
    n_stream = c2.slider("Số giao dịch test chạy live:", 1000, 15000, 5000, step=1000)
    chunk_size = c3.selectbox("Kích thước Chunk:", [1000, 2500, 5000], index=0)
    beta = c4.select_slider("Ngân sách rà soát (beta):", options=[0.0025, 0.005, 0.01], value=0.005)

    run_btn = st.button("Chạy mô phỏng Live", type="primary")
    if not run_btn:
        st.info("Bấm **Chạy mô phỏng Live** để phát trực tiếp luồng giao dịch qua `ARF + Semi-Supervised HST + VerificationController`.")
        return

    with st.spinner("Đang nạp dữ liệu và warm-up nhanh trên 10.000 mẫu lịch sử đã đáo hạn..."):
        warmup_matured, warmup_pending, test_df, feature_cols = load_split_for_replay()
        warm_sub = warmup_matured.iloc[-10000:].reset_index(drop=True)
        test_sub = test_df.iloc[:n_stream].reset_index(drop=True)

        X_warm = warm_sub[feature_cols].to_numpy(dtype=np.float32)
        y_warm = warm_sub["Class"].to_numpy(dtype=np.int32)
        warm_dicts = [dict(zip(feature_cols, row)) for row in X_warm.tolist()]

        test_times = test_sub["Time"].to_numpy(dtype=np.float64)
        test_X = test_sub[feature_cols].to_numpy(dtype=np.float32)
        test_dicts = [dict(zip(feature_cols, row)) for row in test_X.tolist()]
        test_y = test_sub["Class"].to_numpy(dtype=np.int32)

        oml = OnlineFraudModel(n_models=10, seed=42)
        use_hst = strategy == "B4_Hybrid"
        hst = SemiSupervisedHST(seed=42) if use_hst else None

        tail_start = max(0, len(y_warm) - chunk_size)
        tail_p, tail_disc = [], []
        for i in range(len(y_warm)):
            if i >= tail_start:
                p_i = oml.predict_proba_one(warm_dicts[i])
                s_i = hst.score_one(warm_dicts[i]) if (use_hst and hst is not None) else 0.0
                tail_p.append(p_i)
                tail_disc.append(s_i - p_i)
            oml.learn_one(warm_dicts[i], int(y_warm[i]))
            if use_hst and hst is not None:
                hst.learn_one(warm_dicts[i], int(y_warm[i]))

        budget_per_chunk = max(1, int(round(beta * chunk_size)))
        p_arr = np.asarray(tail_p, dtype=np.float64)
        d_arr = np.asarray(tail_disc, dtype=np.float64)
        init_thresh = min(0.95, max(1e-4, float(np.quantile(p_arr, 1.0 - beta))))
        init_top = float(np.quantile(p_arr, max(0.0, 1.0 - (budget_per_chunk * 0.7) / max(1, len(p_arr)))))
        init_disc = float(np.quantile(d_arr, max(0.0, 1.0 - (budget_per_chunk * 0.3) / max(1, len(d_arr)))))

    queue = DelayedLabelQueue(seed=42)
    # Pre-load last 5000 in-flight rows for realism
    pend_sub = warmup_pending.iloc[-5000:].reset_index(drop=True)
    pend_times = pend_sub["Time"].to_numpy(dtype=np.float64)
    pend_dicts = [dict(zip(feature_cols, row)) for row in pend_sub[feature_cols].to_numpy(dtype=np.float32).tolist()]
    pend_y = pend_sub["Class"].to_numpy(dtype=np.int32)
    for idx in range(len(pend_y)):
        queue.push(tx_id=-(idx + 1), t_arrive=float(pend_times[idx]), x=pend_dicts[idx], y_true=int(pend_y[idx]))

    thresh_mgr = CausalQuantileThreshold(alert_rate=beta, initial_threshold=init_thresh)
    ctrl = VerificationController(
        strategy=strategy,
        window_size=chunk_size,
        budget_per_window=budget_per_chunk,
        initial_top_threshold=init_top,
        initial_disc_threshold=init_disc,
        seed=42,
    )

    progress_bar = st.progress(0)
    kpi_box = st.empty()
    chart_box = st.empty()
    log_box = st.empty()

    y_scores = np.zeros(n_stream, dtype=np.float64)
    y_preds = np.zeros(n_stream, dtype=np.int32)
    latencies = np.zeros(n_stream, dtype=np.float64)
    ft_logs: List[Dict[str, object]] = []
    hist_x: List[int] = []
    hist_yield: List[float] = []
    hist_queries: List[int] = []

    update_step = max(100, n_stream // 25)
    chunk_start = 0

    for i in range(n_stream):
        t_now = float(test_times[i])
        x_i = test_dicts[i]
        y_true_i = int(test_y[i])

        for ev in queue.pop_ready(t_now):
            oml.learn_one(ev.x, ev.y_observed)
            if use_hst and hst is not None:
                hst.learn_one(ev.x, ev.y_observed)

        t0 = time.perf_counter()
        p_hat = oml.predict_proba_one(x_i)
        s_hst = hst.score_one(x_i) if (use_hst and hst is not None) else 0.0
        theta = thresh_mgr.get_threshold()
        y_pred_i = 1 if p_hat >= theta else 0
        dec = ctrl.decide(p_hat=p_hat, hst_score=s_hst, decision_threshold=theta)
        latencies[i] = (time.perf_counter() - t0) * 1000.0

        y_scores[i] = p_hat
        y_preds[i] = y_pred_i
        thresh_mgr.record_score(p_hat)

        if dec.should_query:
            if y_true_i == 1:
                ctrl.record_outcome(1)
            ft_logs.append(
                {
                    "tx_id": i,
                    "Time_sec": int(t_now),
                    "p_hat": round(p_hat, 4),
                    "S_HST": round(s_hst, 4),
                    "Delta_disc": round(s_hst - p_hat, 4),
                    "Trigger_Reason": dec.reason,
                    "Actual_Label": "FRAUD (Hit)" if y_true_i == 1 else "Legit",
                }
            )

        queue.push(
            tx_id=i,
            t_arrive=t_now,
            x=x_i,
            y_true=y_true_i,
            is_fast_track=dec.should_query,
            p_hat_at_arrive=p_hat,
        )

        if ((i + 1 - chunk_start) == chunk_size) or (i == n_stream - 1):
            thresh_mgr.end_chunk()
            ctrl.end_window()
            chunk_start = i + 1

        if (i + 1) % update_step == 0 or i == n_stream - 1:
            progress_bar.progress((i + 1) / n_stream)
            hist_x.append(i + 1)
            hist_yield.append(ctrl.verification_yield * 100.0)
            hist_queries.append(ctrl.total_queries)

            with kpi_box.container():
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Đã xử lý", f"{i + 1:,} / {n_stream:,}")
                k2.metric("Lượt Fast-Track đã dùng", f"{ctrl.total_queries}")
                k3.metric("Gian lận bắt được sớm", f"{ctrl.total_true_frauds_found}")
                k4.metric("Verification Yield", f"{ctrl.verification_yield * 100:.2f}%")

            fig, ax1 = plt.subplots(figsize=(9.5, 3.0))
            ax1.plot(hist_x, hist_yield, color="#10b981", marker="o", markersize=3.5, label="Verification Yield (%)")
            ax1.set_xlabel("Số giao dịch đã xử lý")
            ax1.set_ylabel("Verification Yield (%)", color="#10b981")
            ax1.grid(True, linestyle="--", alpha=0.35)
            ax2 = ax1.twinx()
            ax2.plot(hist_x, hist_queries, color="#2563eb", linestyle="--", label="Tổng lượt Fast-Track")
            ax2.set_ylabel("Tổng lượt Fast-Track", color="#2563eb")
            fig.tight_layout()
            chart_box.pyplot(fig)
            plt.close(fig)

            if ft_logs:
                log_box.dataframe(pd.DataFrame(ft_logs[-15:][::-1]), hide_index=True)

    final_m = compute_metrics(
        y_true=test_y,
        y_scores=y_scores,
        y_pred=y_preds,
        k=max(1, int(round(beta * n_stream))),
        latencies_ms=latencies,
        fast_track_queries=ctrl.total_queries,
        fast_track_frauds=ctrl.total_true_frauds_found,
    )
    st.success(
        f"Hoàn tất replay {n_stream:,} giao dịch ({strategy}) | "
        f"AP = {final_m['AP']:.4f} | Pr@k = {final_m['Pr@k']:.4f} | "
        f"Recall = {final_m['Recall']:.4f} | FP = {int(final_m['FP'])} | "
        f"Yield = {final_m['Verification_Yield']*100:.2f}% | "
        f"Mean Latency = {final_m['mean_latency_ms']:.3f} ms/tx"
    )


def main() -> None:
    st.title("OnlineML-ActiveFraud — Selective Verification Dashboard")
    st.caption(
        "Stream-based Active Learning (Semi-Supervised HST–Tree Discordance) dưới môi trường Nhãn trễ vs Batch XGBoost"
    )

    summary, chunks = load_results()
    tab1, tab2, tab3 = st.tabs(
        [
            "1. So sánh 8 Mô hình",
            "2. Quỹ đạo Chunk & Kiến trúc",
            "3. Mô phỏng Luồng Live",
        ]
    )

    with tab1:
        if summary is None:
            st.error("Chưa có `results/ablation_al_summary.csv`. Hãy chạy `python -m experiments.run_benchmark` trước.")
        else:
            render_comparison_tab(summary)

    with tab2:
        if chunks is None:
            st.error("Chưa có `results/chunk_metrics.csv`.")
        else:
            render_chunk_and_arch_tab(chunks)

    with tab3:
        render_live_replay_tab()


if __name__ == "__main__":
    main()