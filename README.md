# OnlineML-ActiveFraud

Khung thực nghiệm **Stream-based Active Learning (Selective Verification)** kết hợp **Semi-Supervised Half-Space Trees (HST–Tree Discordance)** cho bài toán phát hiện gian lận thẻ tín dụng trực tuyến dưới môi trường nhãn trễ (*Delayed Labels*), đối chứng công bằng với Batch XGBoost theo chuẩn *Information Parity*.

## Cấu trúc thư mục

```text
OnlineML-ActiveFraud/
├── src/
│   ├── simulator.py         # Hàng đợi nhãn trễ (Standard + Fast-Track) & chia tập thời gian nhân quả
│   ├── models.py            # Online ARF, Semi-Supervised HST (chỉ học y=0), Batch XGBoost (3 chế độ)
│   ├── active_learning.py   # CausalQuantileThreshold & VerificationController (B0-B4 khóa trần B)
│   └── metrics.py           # AP (PR-AUC), Pr@k, Recall, FP, Verification Yield, Latency (mean/p99)
├── experiments/
│   ├── run_benchmark.py     # Chạy thực nghiệm đối chứng 8 mô hình (5 Online + 3 Batch)
│   └── plot_results.py      # Sinh 4 biểu đồ phân tích từ kết quả CSV
├── tests/
│   └── test_causal_al.py    # 5 unit test kiểm chứng chống Data Leakage, trần ngân sách & parity
├── results/                 # Kết quả ablation_al_summary.csv, chunk_metrics.csv và 4 biểu đồ PNG
├── SPEC_AL_SV.md            # Đặc tả kỹ thuật triển khai
├── REPORT.md                # Báo cáo phân tích kết quả thực nghiệm đầy đủ
└── requirements.txt
## Chạy Dashboard UI tương tác

```powershell
streamlit run dashboard.py
```

