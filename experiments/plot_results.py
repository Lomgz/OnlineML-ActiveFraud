from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_overall_ap_prk(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(summary))
    width = 0.38

    bars1 = ax.bar(x - width / 2, summary["AP"], width, label="AP (PR-AUC)", color="#2563eb")
    bars2 = ax.bar(x + width / 2, summary["Pr@k"], width, label="Pr@k", color="#0ea5e9")

    ax.set_ylabel("Score")
    ax.set_title("Overall Fraud Detection Performance: Online Active Learning vs Batch XGBoost")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=22, ha="right")
    ax.set_ylim(0, min(1.05, max(summary["AP"].max(), summary["Pr@k"].max()) * 1.18 + 0.05))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_verification_yield(summary: pd.DataFrame, out_path: Path) -> None:
    al_df = summary[summary["model"].isin(["B1_Random", "B2_TopRisk", "B3_Margin", "B4_Hybrid"])].copy()
    if al_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(al_df))
    colors = ["#94a3b8", "#f59e0b", "#8b5cf6", "#10b981"]

    bars = ax1.bar(x, al_df["Verification_Yield"] * 100.0, width=0.52, color=colors)
    ax1.set_ylabel("Verification Yield (%)")
    ax1.set_title("Active Learning Query Efficiency (Fixed Verification Budget B)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(al_df["model"])
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    for b, q, f in zip(bars, al_df["FastTrack_Queries"], al_df["FastTrack_Frauds"]):
        h = b.get_height()
        ax1.annotate(
            f"{h:.1f}%\n({int(f)}/{int(q)})",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax1.set_ylim(0, max(10.0, (al_df["Verification_Yield"].max() * 100.0) * 1.25 + 5.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_chunk_trajectory(chunks: pd.DataFrame, out_path: Path) -> None:
    selected = [
        ("B0_NoAL", "#64748b", "--", "o"),
        ("B2_TopRisk", "#f59e0b", "-", "s"),
        ("B4_Hybrid", "#10b981", "-", "D"),
        ("M_Static_Batch", "#ef4444", ":", "^"),
        ("M_Periodic_Batch_AL", "#2563eb", "-", "x"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for model_name, color, ls, marker in selected:
        sub = chunks[chunks["model"] == model_name].sort_values("chunk_id")
        if sub.empty:
            continue
        ax.plot(
            sub["chunk_id"],
            sub["AP"],
            label=model_name,
            color=color,
            linestyle=ls,
            marker=marker,
            linewidth=1.8,
            markersize=4.5,
        )

    ax.set_xlabel("Test Chunk Index (Chronological)")
    ax.set_ylabel("Chunk AP (PR-AUC)")
    ax.set_title("Chunk-by-Chunk AP Trajectory Under Delayed Labels")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_latency_sla(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(summary))
    width = 0.38

    ax.bar(x - width / 2, summary["mean_latency_ms"], width, label="Mean Latency (ms/tx)", color="#0284c7")
    ax.bar(x + width / 2, summary["p99_latency_ms"], width, label="p99 Latency (ms/tx)", color="#f97316")
    ax.axhline(5.0, color="#dc2626", linestyle="--", linewidth=1.2, label="SLA Target (5 ms)")

    ax.set_ylabel("Latency (ms / transaction)")
    ax.set_title("Inference Latency Profile vs Real-Time SLA")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=22, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Active Learning benchmark charts")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    summary = pd.read_csv(res_dir / "ablation_al_summary.csv")
    chunks = pd.read_csv(res_dir / "chunk_metrics.csv")

    plot_overall_ap_prk(summary, res_dir / "fig1_overall_ap_prk.png")
    plot_verification_yield(summary, res_dir / "fig2_verification_yield.png")
    plot_chunk_trajectory(chunks, res_dir / "fig3_chunk_ap_trajectory.png")
    plot_latency_sla(summary, res_dir / "fig4_latency_sla.png")
    print(f"Generated 4 charts in {res_dir}/")


if __name__ == "__main__":
    main()