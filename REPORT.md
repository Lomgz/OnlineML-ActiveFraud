# Báo cáo Tóm tắt Thực nghiệm: Stream Active Learning + HST-Tree Discordance + ADWIN Drift Adaptation

## 1. Kết quả tổng hợp trên 142.404 giao dịch kiểm thử (223 gian lận, beta = 0.5%, seed = 42)

> Trần lý thuyết của `Pr@k` tại $k = 712$ là $223 / 712 = \mathbf{0.3132}$.

| Mô hình | Nhóm | AP (PR-AUC) | Pr@k (k=712) | Recall | Precision | TP | FP | Queries | Frauds | Yield | ADWIN Drifts | Mean Lat (ms) | p99 Lat (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0_NoAL` | Online ARF | 0.4804 | 0.1840 | 0.5785 | 0.0853 | 129 | 1,383 | 0 | 0 | 0.00% | 7 | 0.029 | 0.035 |
| `B1_Random` | Online ARF | 0.4383 | 0.2051 | 0.5695 | 0.1203 | 127 | 929 | 714 | 0 | 0.00% | 9 | 0.032 | 0.036 |
| `B2_TopRisk` | Online ARF | 0.4075 | 0.1952 | 0.5471 | 0.1032 | 122 | 1,060 | 709 | 78 | 11.00% | 8 | 0.033 | 0.038 |
| `B3_Margin` | Online ARF | 0.5701 | 0.2289 | 0.5964 | 0.0331 | 133 | 3,890 | 725 | 13 | 1.79% | 12 | 0.031 | 0.036 |
| **`B4_Hybrid`** | **Online ARF** | **0.6058** | **0.2303** | **0.6233** | **0.1466** | **139** | **809** | **710** | **75** | **10.56%** | **7** | **0.105** | **0.122** |
| `M_Static_Batch` | Batch XGBoost | 0.7924 | 0.2640 | 0.8296 | 0.2873 | 185 | 459 | 0 | 0 | — | 0 | 0.460 | 1.193 |
| `M_Periodic_Batch` | Batch XGBoost | 0.7888 | 0.2669 | 0.8430 | 0.2640 | 188 | 524 | 0 | 0 | — | 0 | 0.382 | 0.575 |
| `M_Periodic_Batch_AL` | Batch XGBoost | 0.7796 | 0.2669 | 0.8430 | 0.2773 | 188 | 490 | 0 | 0 | — | 0 | 0.409 | 0.615 |

## 2. Nhận định cốt lõi

- **`B4_Hybrid` dẫn đầu nhóm Online:** `AP = 0.6058` (+26.1% so với `B0_NoAL`), thu hẹp `40.2%` khoảng cách AP với `M_Static_Batch` (`0.7924`) và `55.9%` khoảng cách `Pr@k` với `M_Periodic_Batch` (`0.2669`), giữ `FP` thấp nhất (`809`) và `Precision` cao nhất (`14.66%`) trong 5 mô hình Online.
- **Độ trễ thực tế:** `B4_Hybrid` đạt `0.105 ms/tx` (`p99 = 0.122 ms`), nhanh hơn khoảng `4x` so với Batch XGBoost đơn giao dịch (`0.38–0.46 ms`) và đạt chuẩn SLA `< 5 ms`.
- **Phát hiện thực nghiệm về ADWIN trên dữ liệu mất cân bằng cực đoan:** Chuỗi lỗi `0-1` kích hoạt ADWIN `7–12` lần trên tập ULB. Ở `B0_NoAL` (`FP = 1,383`) và `B3_Margin` (`FP = 3,890`), việc tái hiệu chỉnh ngưỡng ngắn hạn gây vòng lặp tự kích hoạt FP–ADWIN; ngược lại, `B4_Hybrid` kiềm chế tốt `FP` (`809`) nhờ kết hợp Top-Risk và Semi-Supervised HST Discordance.
- **Biểu đồ đối chiếu:** Xem `results/fig1_overall_ap_prk.png`, `results/fig2_verification_yield.png`, `results/fig3_chunk_ap_trajectory.png`, `results/fig4_latency_sla.png`.
