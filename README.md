# OnlineML-ActiveFraud

Khung thực nghiệm **Stream-based Active Learning (Selective Verification)** kết hợp **Semi-Supervised Half-Space Trees (HST–Tree Discordance)** và **ADWIN Drift Adaptation** cho bài toán phát hiện gian lận thẻ tín dụng trực tuyến dưới môi trường nhãn trễ (*Delayed Labels*), đối chứng công bằng với Batch XGBoost theo chuẩn *Information Parity*.

## Cấu trúc thư mục

```text
OnlineML-ActiveFraud/
├── src/
│   ├── __init__.py
│   ├── simulator.py         # Hàng đợi nhãn trễ (Standard + Fast-Track) & chia tập thời gian nhân quả
│   ├── models.py            # Online ARF (n_models=10), Semi-Supervised HST (chỉ học y=0), Batch XGBoost
│   ├── active_learning.py   # CausalQuantileThreshold & VerificationController (B0-B4 khóa trần B)
│   ├── drift.py             # ADWINDriftManager nhận chuỗi 0-1 prediction error tại t_label + cooldown boost
│   └── metrics.py           # AP (PR-AUC), Pr@k, Recall, Precision, TP, FP, Verification Yield, Latency
├── experiments/
│   ├── run_benchmark.py     # Chạy thực nghiệm đối chứng 8 mô hình (5 Online + 3 Batch)
│   └── plot_results.py      # Sinh 4 biểu đồ phân tích từ kết quả CSV
├── tests/
│   ├── test_causal_al.py    # 5 unit tests chống Data Leakage, trần ngân sách & Information Parity
│   └── test_adwin_drift.py  # 3 unit tests cho ADWIN error-stream, cooldown boost & tái hiệu chỉnh ngưỡng
├── results/                 # ablation_al_summary.csv, chunk_metrics.csv và 4 biểu đồ PNG
├── dashboard.py             # Streamlit + Plotly Payment Edge Monitor (Live Stream + Benchmark)
├── SPEC_AL_SV.md            # Đặc tả kỹ thuật triển khai
├── REPORT.md                # Báo cáo tóm tắt thực nghiệm
└── requirements.txt
```

## Cơ chế cốt lõi (Online ML + Delayed Labels + ADWIN)

1. **Strict Causality & Delayed Labels (`src/simulator.py`):** Chia tập 50/50 theo cột `Time` (loại `Time` khỏi tập đặc trưng). Nhãn chuẩn trễ $D = 21.600\text{s}$ (6 giờ, kèm `5%` gian lận không báo cáo); kênh *Fast-Track* của Active Learning trả nhãn thực sau `300s` (5 phút). Mô hình chỉ gọi `learn_one` khi nhãn đáo hạn (`t_label <= t_now`).
2. **B4_Hybrid Selective Verification (`src/active_learning.py` & `src/models.py`):** Chia ngân sách cố định $\beta = 0.5\%$ ($B = 25$ lượt / $5.000$ giao dịch) thành `70%` Top-Risk ($\hat{p}_{\text{Tree}}$) và `30%` bất đồng *HST–Tree Discordance* ($\Delta_{\text{disc}} = S_{\text{HST}} - \hat{p}_{\text{Tree}}$), trong đó `SemiSupervisedHST` chỉ cập nhật từ mẫu xác nhận hợp lệ (`y == 0`).
3. **ADWIN Error-Stream & Drift Response (`src/drift.py`):** `ADWINDriftManager` nhận chuỗi lỗi nhị phân `int(y_pred_at_arrive != y_observed)` tại thời điểm nhãn đáo hạn. Khi `adwin.drift_detected` kích hoạt: ghi log sự kiện drift, tăng tỷ trọng khám phá Discordance của `B4_Hybrid` từ `30%` lên `50%` trong cửa sổ cooldown (vẫn giữ trần cứng $B$), và tái hiệu chỉnh `CausalQuantileThreshold` từ lịch sử chunk đã hoàn tất.

## Cài đặt & Chạy kiểm chứng

```powershell
# Khuyến nghị Python 3.11 / 3.12
python -m pip install -r requirements.txt

# Chạy 8 unit tests (5 causal AL + 3 ADWIN drift)
pytest -v

# Chạy thực nghiệm đối chứng 8 mô hình (ULB creditcard.csv)
python -m experiments.run_benchmark --seed 42 --beta 0.005

# Sinh 4 biểu đồ phân tích
python -m experiments.plot_results

# Mở Streamlit Payment Edge Monitor (Live Stream + Benchmark)
streamlit run dashboard.py
```
