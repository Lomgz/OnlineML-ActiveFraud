from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.active_learning import CausalQuantileThreshold, VerificationController
from src.drift import ADWINDriftManager
from src.metrics import compute_metrics
from src.models import OnlineFraudModel, SemiSupervisedHST
from src.simulator import DelayedLabelQueue, prepare_temporal_split

RESULTS_DIR = ROOT_DIR / "results"
DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"

st.set_page_config(
    page_title="Payment Edge Fraud Monitor — OnlineML & Active Learning",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Anti-dimming / anti-flicker CSS (aligned with Membau/OnlineML-Fraud-Detector)
NO_DIM_CSS = """
<style>
html, body, [class*="css"], [class*="st-"], .stApp,
.stApp *,
div[data-testid="stAppViewContainer"],
div[data-testid="stAppViewContainer"] *,
section.main,
section.main *,
div.block-container,
div.block-container *,
[data-testid="stFragment"],
[data-testid="stFragment"] *,
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlock"] *,
[data-testid="stElementContainer"],
[data-testid="stElementContainer"] *,
div[data-testid="stPlotlyChart"],
div[data-testid="stPlotlyChart"] * {
    opacity: 1 !important;
    filter: none !important;
    transition: none !important;
    animation: none !important;
}
[data-testid="stStatusWidget"],
div[data-testid="stStatusWidget"],
div[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}
.js-plotly-plot,
.js-plotly-plot .plotly,
.js-plotly-plot .plotly .main-svg {
    transition: none !important;
    opacity: 1 !important;
}
</style>
"""
st.markdown(NO_DIM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_benchmark_results():
    summary_path = RESULTS_DIR / "ablation_al_summary.csv"
    chunk_path = RESULTS_DIR / "chunk_metrics.csv"
    if not summary_path.exists() or not chunk_path.exists():
        return None, None
    return pd.read_csv(summary_path), pd.read_csv(chunk_path)


@st.cache_data
def load_split_for_replay(confirm_window_sec: float = 21600.0):
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    return prepare_temporal_split(
        df,
        warmup_ratio=0.5,
        confirm_window_sec=confirm_window_sec,
    )


def render_live_monitor_tab(
    strategy: str,
    beta: float,
    window_size: int,
    confirm_window_sec: float,
    fast_track_delay_sec: float,
    adwin_delta: float,
    enable_adwin_action: bool,
    n_stream: int,
) -> None:
    split_data = load_split_for_replay(confirm_window_sec=confirm_window_sec)
    if split_data is None:
        st.warning(f"Khong tim thay du lieu tai `{DATA_PATH}`. Vui long dat `creditcard.csv` vao `data/`.")
        return

    warmup_matured, warmup_pending, test_df, feature_cols = split_data
    st.caption(
        f"Streaming Engine: `OnlineML-ActiveFraud` | Strategy: `{strategy}` | "
        f"Alert Budget beta: `{beta:.3%}` | Window W: `{window_size:,}` | "
        f"Standard Delay D: `{int(confirm_window_sec)}s` | Fast-Track: `{int(fast_track_delay_sec)}s`"
    )

    start_btn = st.button("Khoi chay Stream Monitor (Delayed Labels + Active Learning + ADWIN)", type="primary")
    if not start_btn:
        st.info("Nhan nut ben tren de mo phong luong giao dich ULB thoi gian thuc (Warm-up -> DelayedLabelQueue -> ADWIN).")
        return

    warm_sample = warmup_matured.tail(15000).reset_index(drop=True)
    stream_sample = test_df.iloc[:n_stream].reset_index(drop=True)
    budget_per_win = max(1, int(round(beta * window_size)))

    oml = OnlineFraudModel(n_models=10, seed=42)
    hst = SemiSupervisedHST(seed=42)
    use_hst = strategy == "B4_Hybrid"

    warm_tail_p: List[float] = []
    warm_tail_disc: List[float] = []
    tail_start = max(0, len(warm_sample) - 2000)
    prog = st.progress(0, text="Dang khoi tao Warm-up ARF + Semi-Supervised HST...")

    for i, row in warm_sample.iterrows():
        x_i = {c: float(row[c]) for c in feature_cols}
        y_i = int(row["Class"])
        if i >= tail_start:
            p_i = oml.predict_proba_one(x_i)
            s_i = hst.score_one(x_i)
            warm_tail_p.append(p_i)
            warm_tail_disc.append(s_i - p_i)
        oml.learn_one(x_i, y_i)
        hst.learn_one(x_i, y_i)

    p_arr = np.asarray(warm_tail_p, dtype=np.float64)
    d_arr = np.asarray(warm_tail_disc, dtype=np.float64)
    init_thresh = min(0.95, max(1e-4, float(np.quantile(p_arr, 1.0 - beta))))
    top_rate = max(1e-4, (budget_per_win * 0.70) / max(1, len(p_arr)))
    disc_rate = max(1e-4, (budget_per_win * 0.30) / max(1, len(d_arr)))

    thresh_mgr = CausalQuantileThreshold(alert_rate=beta, initial_threshold=init_thresh)
    ctrl = VerificationController(
        strategy=strategy,
        window_size=window_size,
        budget_per_window=budget_per_win,
        hybrid_top_ratio=0.70,
        initial_top_threshold=float(np.quantile(p_arr, max(0.0, 1.0 - top_rate))),
        initial_disc_threshold=float(np.quantile(d_arr, max(0.0, 1.0 - disc_rate))),
        seed=42,
    )
    drift_mgr = ADWINDriftManager(
        delta=adwin_delta,
        cooldown_duration=300,
        base_disc_ratio=0.30,
        boosted_disc_ratio=0.50,
    )
    queue = DelayedLabelQueue(
        confirm_window_sec=confirm_window_sec,
        min_fraud_delay_sec=1800.0,
        fast_track_delay_sec=fast_track_delay_sec,
        unreported_fraud_rate=0.05,
        seed=42,
    )

    for idx, row in warmup_pending.iterrows():
        queue.push(
            tx_id=-(idx + 1),
            t_arrive=float(row["Time"]),
            x={c: float(row[c]) for c in feature_cols},
            y_true=int(row["Class"]),
            is_fast_track=False,
        )

    y_true_all: List[int] = []
    y_score_all: List[float] = []
    y_pred_all: List[int] = []
    lat_all: List[float] = []
    stream_rows: List[Dict[str, object]] = []
    queue_depths: List[int] = []

    allow_cnt, block_cnt, ft_cnt = 0, 0, 0

    for i, row in stream_sample.iterrows():
        t_now = float(row["Time"])
        x_i = {c: float(row[c]) for c in feature_cols}
        y_true_i = int(row["Class"])

        for ev in queue.pop_ready(t_now):
            oml.learn_one(ev.x, ev.y_observed)
            if use_hst:
                hst.learn_one(ev.x, ev.y_observed)
            if ev.tx_id >= 0:
                fired = drift_mgr.update(
                    y_pred=ev.y_pred_at_arrive,
                    y_true=ev.y_observed,
                    current_time=t_now,
                )
                if fired and enable_adwin_action:
                    thresh_mgr.recalibrate_on_drift()
                if enable_adwin_action and strategy == "B4_Hybrid":
                    ctrl.set_discordance_ratio(drift_mgr.get_current_disc_ratio())

        ts = time.perf_counter()
        p_hat = oml.predict_proba_one(x_i)
        s_hst = hst.score_one(x_i) if use_hst else 0.0
        theta = thresh_mgr.get_threshold()
        y_pred_i = 1 if p_hat >= theta else 0
        dec = ctrl.decide(p_hat=p_hat, hst_score=s_hst, decision_threshold=theta)
        lat_ms = (time.perf_counter() - ts) * 1000.0

        if dec.should_query:
            action = "FAST_TRACK"
            ft_cnt += 1
            if y_true_i == 1:
                ctrl.record_outcome(1)
        elif y_pred_i == 1:
            action = "BLOCK"
            block_cnt += 1
        else:
            action = "ALLOW"
            allow_cnt += 1

        y_true_all.append(y_true_i)
        y_score_all.append(p_hat)
        y_pred_all.append(y_pred_i)
        lat_all.append(lat_ms)
        thresh_mgr.record_score(p_hat)
        queue_depths.append(queue.pending_count())

        stream_rows.append(
            {
                "tx_id": i,
                "t_arrive": t_now,
                "p_hat": p_hat,
                "S_HST": s_hst,
                "Delta_disc": max(0.0, s_hst - p_hat),
                "theta": theta,
                "action": action,
                "y_true": y_true_i,
            }
        )

        queue.push(
            tx_id=i,
            t_arrive=t_now,
            x=x_i,
            y_true=y_true_i,
            is_fast_track=dec.should_query,
            p_hat_at_arrive=p_hat,
            y_pred_at_arrive=y_pred_i,
        )

        if (i + 1) % window_size == 0:
            thresh_mgr.end_chunk()
            ctrl.end_window()

        if (i + 1) % max(1, n_stream // 20) == 0 or i == n_stream - 1:
            prog.progress(int((i + 1) / n_stream * 100), text=f"Dang xu ly {i + 1:,}/{n_stream:,} giao dich...")

    prog.empty()

    k_eval = max(1, int(round(beta * n_stream)))
    m = compute_metrics(
        y_true=y_true_all,
        y_scores=y_score_all,
        y_pred=y_pred_all,
        k=k_eval,
        latencies_ms=lat_all,
        fast_track_queries=ctrl.total_queries,
        fast_track_frauds=ctrl.total_true_frauds_found,
    )

    # 6 KPI Metric Cards (Membau/OnlineML-Fraud-Detector layout)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Tong Giao Dich", f"{n_stream:,}")
    k2.metric("PR-AUC (AP)", f"{m['AP']:.4f}")
    k3.metric(f"Pr@{k_eval}", f"{m['Pr@k']:.4f}")
    k4.metric("Precision / Recall", f"{m['Precision']:.3f} / {m['Recall']:.3f}")
    k5.metric("Verification Yield", f"{m['Verification_Yield']*100:.2f}%", f"{int(m['FastTrack_Frauds'])}/{int(m['FastTrack_Queries'])}")
    k6.metric("ADWIN Drifts", f"{len(drift_mgr.drift_events)}", f"Cooldown: {'ON' if drift_mgr.is_in_cooldown() else 'OFF'}")

    st.divider()
    sdf = pd.DataFrame(stream_rows)
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("Xac Suat Gian Lan (ARF) & Bat Thuong HST Theo Thoi Gian")
        tail_df = sdf.tail(400)
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=tail_df["tx_id"], y=tail_df["p_hat"], name="p_hat (ARF)", line=dict(color="#58A6FF", width=1.8)))
        if use_hst:
            fig_line.add_trace(go.Scatter(x=tail_df["tx_id"], y=tail_df["S_HST"], name="S_HST", line=dict(color="#3FB950", width=1.4)))
        fig_line.add_trace(go.Scatter(x=tail_df["tx_id"], y=tail_df["theta"], name="Causal Threshold", line=dict(color="#FACC15", dash="dash")))
        fig_line.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_line, use_container_width=True)

    with col_r:
        st.subheader("Ty Le ALLOW / BLOCK / FAST_TRACK")
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=["ALLOW", "BLOCK", "FAST_TRACK"],
                    values=[allow_cnt, block_cnt, ft_cnt],
                    hole=0.45,
                    marker_colors=["#238636", "#DA3633", "#D29922"],
                )
            ]
        )
        fig_pie.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    c_cm, c_q = st.columns(2)
    with c_cm:
        tp = int(m["TP"])
        fp = int(m["FP"])
        fn = int(sum(y_true_all)) - tp
        tn = len(y_true_all) - tp - fp - fn
        cm = [[tn, fp], [fn, tp]]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Pred 0 (Allow)", "Pred 1 (Block)"],
                y=["True 0 (Legit)", "True 1 (Fraud)"],
                colorscale="Blues",
                showscale=False,
                text=cm,
                texttemplate="%{text}",
            )
        )
        fig_cm.update_layout(template="plotly_dark", title="Cumulative Confusion Matrix", height=270, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_q:
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=sdf["tx_id"], y=queue_depths, name="Pending Labels", fill="tozeroy", line=dict(color="#D29922")))
        fig_q.update_layout(template="plotly_dark", title="DelayedLabelQueue Backlog (In-Flight Labels)", height=270, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_q, use_container_width=True)

    if drift_mgr.drift_events:
        st.subheader("ADWIN Drift Events Log (0-1 Error Stream)")
        st.dataframe(pd.DataFrame(drift_mgr.drift_events), use_container_width=True, hide_index=True)

    st.subheader("Nhat Ky Giao Dich Gan Nhat (Audit Log)")
    audit_df = sdf.sort_values("tx_id", ascending=False).head(100)
    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "p_hat": st.column_config.ProgressColumn("P(fraud) ARF", min_value=0.0, max_value=1.0, format="%.4f"),
            "S_HST": st.column_config.ProgressColumn("S_HST Anomaly", min_value=0.0, max_value=1.0, format="%.4f"),
            "Delta_disc": st.column_config.ProgressColumn("Discordance", min_value=0.0, max_value=1.0, format="%.4f"),
            "action": st.column_config.TextColumn("Quyet Dinh"),
        },
    )


def render_benchmark_tab(summary: pd.DataFrame, chunks: pd.DataFrame) -> None:
    b0 = summary.loc[summary["model"] == "B0_NoAL"].iloc[0]
    b4 = summary.loc[summary["model"] == "B4_Hybrid"].iloc[0]
    m_static = summary.loc[summary["model"] == "M_Static_Batch"].iloc[0]

    ap_gain = ((b4["AP"] - b0["AP"]) / max(1e-9, b0["AP"])) * 100.0
    gap_closed = ((b4["AP"] - b0["AP"]) / max(1e-9, m_static["AP"] - b0["AP"])) * 100.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("B4_Hybrid AP", f"{b4['AP']:.4f}", f"+{ap_gain:.1f}% vs B0")
    k2.metric("B4_Hybrid Pr@k", f"{b4['Pr@k']:.4f}", f"vs {b0['Pr@k']:.4f} (B0)")
    k3.metric("B4 Recall / Prec", f"{b4['Recall']:.3f} / {b4['Precision']:.3f}")
    k4.metric("Verification Yield", f"{b4['Verification_Yield']*100:.2f}%", f"{int(b4['FastTrack_Frauds'])}/{int(b4['FastTrack_Queries'])}")
    k5.metric("Gap Closed vs XGB", f"{gap_closed:.1f}%")
    k6.metric("Latency (B4)", f"{b4['mean_latency_ms']:.3f} ms", f"p99: {b4['p99_latency_ms']:.3f} ms")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(
            summary,
            x="model",
            y=["AP", "Pr@k"],
            barmode="group",
            template="plotly_dark",
            title="Online Active Learning vs Batch XGBoost (AP & Pr@k)",
            height=360,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        al_df = summary[summary["FastTrack_Queries"] > 0]
        fig_yield = px.bar(
            al_df,
            x="model",
            y=al_df["Verification_Yield"] * 100.0,
            text=al_df.apply(lambda r: f"{r['Verification_Yield']*100:.2f}% ({int(r['FastTrack_Frauds'])}/{int(r['FastTrack_Queries'])})", axis=1),
            template="plotly_dark",
            title="Active Learning Verification Yield (%)",
            labels={"y": "Yield (%)"},
            height=360,
        )
        st.plotly_chart(fig_yield, use_container_width=True)

    selected = st.multiselect(
        "Chon mo hinh xem quy dao Chunk AP:",
        options=summary["model"].tolist(),
        default=["B0_NoAL", "B2_TopRisk", "B4_Hybrid", "M_Static_Batch", "M_Periodic_Batch_AL"],
    )
    if selected:
        sub = chunks[chunks["model"].isin(selected)]
        fig_traj = px.line(
            sub,
            x="chunk_id",
            y="AP",
            color="model",
            markers=True,
            template="plotly_dark",
            title="Chunk-by-Chunk AP Trajectory (29 Chunks, 142.404 Test Transactions)",
            height=360,
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    st.subheader("Bang Tong Hop Benchmark (142.404 Giao Dich Kiem Thu)")
    st.dataframe(summary, use_container_width=True, hide_index=True)


with st.sidebar:
    st.header("Cau Hinh Cong Bien & Active Learning")
    st.caption("OnlineML-ActiveFraud | Delayed Labels + ADWIN")
    strategy = st.selectbox("Chien luoc Active Learning", ["B4_Hybrid", "B2_TopRisk", "B3_Margin", "B1_Random", "B0_NoAL"], index=0)
    beta = st.slider("Ngan sach ra soat (beta)", 0.0025, 0.0200, 0.0050, 0.0025, format="%.4f")
    window_size = st.slider("Cua so ngan sach (W)", 1000, 5000, 5000, 1000)
    confirm_window_sec = st.slider("Do tre nhan chuan D (giay)", 3600, 21600, 21600, 3600)
    fast_track_delay_sec = st.slider("Do tre Fast-Track d_fast (giay)", 60, 900, 300, 60)
    adwin_delta = st.select_slider("ADWIN delta (0-1 error stream)", options=[0.0005, 0.001, 0.002, 0.005, 0.01], value=0.002)
    enable_adwin_action = st.checkbox("Kich hoat phan ung ADWIN (Boost Discordance 30%->50%)", value=True)
    n_stream = st.slider("So giao dich mo phong Live", 2000, 30000, 8000, 2000)

st.title("Payment Edge Fraud Monitor — Active Learning & ADWIN")
tab1, tab2 = st.tabs(["Live Stream Monitor (AL + Delayed Labels + ADWIN)", "Benchmark Results (142.404 Tx)"])

with tab1:
    render_live_monitor_tab(
        strategy=strategy,
        beta=beta,
        window_size=window_size,
        confirm_window_sec=float(confirm_window_sec),
        fast_track_delay_sec=float(fast_track_delay_sec),
        adwin_delta=float(adwin_delta),
        enable_adwin_action=enable_adwin_action,
        n_stream=n_stream,
    )

with tab2:
    summary_df, chunk_df = load_benchmark_results()
    if summary_df is None or chunk_df is None:
        st.warning("Chua co file ket qua trong `results/`. Hay chay `python experiments/run_benchmark.py`.")
    else:
        render_benchmark_tab(summary_df, chunk_df)
