# SPEC TRIỂN KHAI: STREAM ACTIVE LEARNING & SELECTIVE VERIFICATION (AL-SV)

**Phạm vi:** Nhánh nghiên cứu mới (repo/nhánh sạch tách khỏi bản lưu trữ `Lomgz`)  
**Dữ liệu mục tiêu:** ULB Credit Card Fraud (`creditcard.csv`)  
**Mục tiêu bàn giao:** Pipeline sạch, bộ thực nghiệm đối chứng công bằng (không rò rỉ tương lai), kiểm thử tự động (`pytest`), và bảng số liệu + biểu đồ phục vụ báo cáo.

---

## 1. Chốt Phạm Vi & Hiệu Chỉnh So Với Bản Nháp Trước

Nhánh mới giữ nguyên trục nghiên cứu cốt lõi từ bản thảo `nghien_cuu_active_learning_selective_verification_oml.md` (Học chủ động có ràng buộc ngân sách $\beta$, chỉ số xung đột `HST-Tree Discordance`, và thước đo `Verification Yield`), đồng thời khóa chặt 4 điểm điều chỉnh kỹ thuật trước khi viết code:

1. **Loại bỏ `CCD-PR`:** Tránh trùng lặp cơ chế `PER` đã có trên repo cũ và tránh suy giảm hiệu năng đã ghi nhận ở cấu hình `M7`. Toàn bộ đóng góp tập trung vào bộ điều khiển **Selective Verification**.
2. **Đồng bộ lý thuyết và mã nguồn:** Cài đặt đầy đủ tầng `Half-Space Trees` bán giám sát, chỉ số xung đột $\Delta_{\mathrm{disc}}$ và ngưỡng thích nghi động theo ngân sách (không dùng quy tắc chặn cứng `abs(prob - 0.70) <= 0.08`).
3. **Đảm bảo tính công bằng thông tin (Information Parity) và tính nhân quả:** Mọi chiến lược Active Learning đều dùng chung một trần ngân sách $B$ và cùng độ trễ Fast-Track $d_{\mathrm{fast}}$; mọi quyết định xin nhãn đều chỉ dùng thống kê quá khứ (tuyệt đối không nhìn trước mẫu tương lai trong cửa sổ).
4. **Mở rộng quy mô chạy:** Không đánh giá trên $6.000$ dòng (quá ít mẫu gian lận); chạy tối thiểu $30.000$ dòng cho bản kiểm tra nhanh và toàn bộ $50\%$ Suffix ($142.404$ giao dịch) cho bảng số liệu báo cáo cuối.

---

## 2. Cấu Trúc Workspace Sạch (Clean Layout)

Dời toàn bộ file phân tích nháp và kết quả cũ ra khỏi thư mục làm việc chính (hoặc tạo repo mới với cấu trúc tối giản sau):

```text
OnlineML-ActiveVerification/
├── data/
│   └── creditcard.csv                # Không commit lên git (.gitignore)
├── src/
│   ├── __init__.py
│   ├── stream_simulator.py           # Chia tập 50/50 theo thời gian + Hàng đợi nhãn trễ & Fast-Track
│   ├── models.py                     # Lớp bọc ARF/HAT + Semi-Supervised HalfSpaceTrees (kèm MinMaxScaler)
│   ├── active_learning.py            # Bộ điều khiển ngân sách B và 5 chiến lược chọn mẫu nhân quả
│   └── metrics.py                    # PR-AUC, Pr@k, Recall, FP, Verification Yield, Latency SLA
├── experiments/
│   ├── run_benchmark.py              # Script chạy toàn bộ 7 cấu hình đối chứng + xuất CSV/JSON
│   └── plot_results.py               # Sinh biểu đồ so sánh (PR-AUC, Pr@k, Yield, Delay Reduction)
├── tests/
│   └── test_al_pipeline.py           # Pytest bắt buộc: Causality, Budget Cap, HST Purity, No Time Leakage
├── results/                          # Chỉ chứa CSV, PNG và summary mới nhất của lần chạy chốt
├── SPEC_AL_SV.md                     # Tài liệu đặc tả này
├── requirements.txt
└── README.md
```

---

## 3. Giao Thức Luồng Dữ Liệu & Độ Trễ Nhãn (Stream & Delay Protocol)

### 3.1. Chia tập và xử lý cột `Time`
* **Chia theo thứ tự thời gian (Chronological Split):** $50\%$ Prefix ($142.403$ dòng đầu) dùng để khởi tạo/warm-up mô hình; $50\%$ Suffix ($142.404$ dòng sau) dùng để đánh giá streaming.
* **Vai trò cột `Time`:** Giữ `Time` (đơn vị giây) làm **đồng hồ luồng (metadata)** để điều phối độ trễ nhãn và cắt chunk đánh giá; **loại bỏ hoàn toàn `Time`** khỏi vectơ đặc trưng $\mathbf{x}_t$ đưa vào mô hình (`V1` đến `V28`, `Amount`).
* **Cắt vùng chưa đáo hạn cho Static Batch:** Mô hình tĩnh `B0_BatchStatic` chỉ được huấn luyện trên các giao dịch thuộc $50\%$ Prefix có thời điểm nhận nhãn $\tau_{\mathrm{label}} \le t_0$ (cắt bỏ đoạn $(t_0 - D, t_0]$) để chống rò rỉ dữ liệu.

### 3.2. Hai hàng đợi nhãn song song (Zero Leakage)
Tại thời điểm $\tau_t$, hệ thống chỉ gọi `predict_proba_one` (tuyệt đối không gọi `learn_one`). Nhãn thực tế $y_t$ chỉ đi vào `learn_one` khi đồng hồ luồng đạt $\tau_{\mathrm{label}}(t)$:

* **Hàng đợi nhãn trễ tự nhiên ($a_t = 0$):** Tham chiếu khung `Protocol_v2` trên ULB với cửa sổ xác nhận $D = 10.800\text{ s}$ ($3\text{ giờ}$):
  * Giao dịch hợp lệ ($y_t = 0$): xác nhận tại $\tau_{\mathrm{label}} = \tau_t + D$.
  * Giao dịch gian lận ($y_t = 1$): $95\%$ có độ trễ thực ngẫu nhiên theo phân phối Log-Normal cắt ngọn trong $(0, D]$ (trung vị $\approx 0.3D$); hỗ trợ tham số tùy chọn `unreported_fraud_rate` $\in \{0.0, 0.05\}$ để kiểm thử khả năng chịu nhiễu nhãn.
* **Hàng đợi Fast-Track Verification ($a_t = 1$):** Nếu giao dịch được chọn bởi Active Learning (và còn ngân sách), độ trễ nhãn rút ngắn xuống $d_{\mathrm{fast}} = 900\text{ s}$ ($15\text{ phút}$), trả về nhãn xác minh chính xác $y_t$ tại $\tau_{\mathrm{label}} = \min(\tau_t + d_{\mathrm{fast}}, \tau_{\mathrm{natural}})$.

---

## 4. Đặc Tả Thuật Toán & Công Thức Cốt Lõi

### 4.1. Kiến trúc Hybrid hai tầng
* **Tầng 1 (Semi-Supervised Anomaly Detector):** Đường ống `MinMaxScaler() | HalfSpaceTrees(n_trees=10, height=6, window_size=250, seed=42)` cho ra điểm bất thường chuẩn hóa $S_{\mathrm{HST}}(\mathbf{x}_t) \in [0, 1]$.
  * **Quy tắc cập nhật bắt buộc:** Chỉ gọi `hst.learn_one(x_i)` khi nhận phản hồi xác nhận giao dịch sạch ($y_i = 0$). Không bao giờ cập nhật HST trên nhãn gian lận ($y_i = 1$) để tránh hấp thụ mẫu gian lận vào vùng bình thường.
* **Tầng 2 (Supervised Online Classifier):** Sử dụng `ARFClassifier(n_models=5, seed=42)` (hoặc cấu hình CLI sang `HoeffdingAdaptiveTreeClassifier` để so sánh tốc độ). Cập nhật bằng `learn_one(x, y)` chuẩn của `river`.

### 4.2. Chỉ số Xung đột Tri thức (HST-Tree Discordance)
Với mỗi giao dịch $\mathbf{x}_t$ đến cổng biên, hệ thống tính:

$$
\Delta_{\mathrm{disc}}(\mathbf{x}_t) = S_{\mathrm{HST}}(\mathbf{x}_t) - \hat{p}_{\mathrm{Tree}}(\mathbf{x}_t)
$$

Giá trị $\Delta_{\mathrm{disc}}(\mathbf{x}_t)$ cao cho thấy tầng không giám sát phát hiện bất thường rõ rệt trong khi cây phân loại đang tự tin gán là giao dịch sạch (điểm mù Zero-day Fraud / Concept Drift mới).

### 4.3. Ngưỡng quyết định động (Khắc phục bẫy Recall thấp)
Thay vì chốt cứng ngưỡng chặn `0.70`, hệ thống đánh giá trên từng chunk $3\text{ giờ}$ (hoặc cửa sổ $W$ mẫu) và cập nhật ngưỡng quyết định $\theta_c$ bằng **phân vị (quantile)** của điểm rủi ro trên chunk liền trước sao cho khớp ngân sách cảnh báo $k = \alpha \times \text{fraud\_rate}_{\mathrm{prev}} \times N_{\mathrm{chunk}}$ với $\alpha = 1.0$.

### 4.4. Bộ điều khiển Ngân sách & Chọn mẫu Nhân quả (Causal Budget Controller)
Cố định trần ngân sách xác minh theo từng cửa sổ trượt kích thước $W = 2.000$ giao dịch với tỷ lệ $\beta = 0.005$ ($0.5\%$, tương đương $B = \lfloor \beta \cdot W \rfloor = 10$ lượt xác minh mỗi cửa sổ; hỗ trợ quét $\beta \in \{0.0025, 0.005, 0.01\}$):

* **Điều kiện trần cứng:** Trong mỗi cửa sổ $W$, chỉ cho phép $a_t = 1$ nếu số lượt đã dùng $u_w < B$.
* **Cơ chế ra quyết định nhân quả (không nhìn trước tương lai):** Duy trì hàng đợi trượt `deque(maxlen=W)` lưu lịch sử điểm rủi ro $\hat{p}$ và điểm xung đột $\Delta_{\mathrm{disc}}$ của các mẫu đã đi qua để tính ngưỡng phân vị tức thời:
  * **Top-Risk (`B2`):** Kích hoạt khi $\hat{p}_t \ge Q_{1 - \beta}(\mathcal{H}_{\hat{p}})$.
  * **Adaptive Margin (`B3`):** Kích hoạt khi $|\hat{p}_t - \theta_t| \le \delta_t$, trong đó $\delta_t$ cập nhật theo Žliobaitė et al. (2014) với tốc độ $\gamma = 0.01$:
    $$
    \delta_{t+1} =
    \begin{cases}
    \delta_t (1 - \gamma), & \text{nếu } \hat{\beta}_t > \beta, \\
    \delta_t (1 + \gamma), & \text{nếu } \hat{\beta}_t < \beta.
    \end{cases}
    $$
  * **Discordance (`B4`):** Kích hoạt khi $\Delta_{\mathrm{disc}}(\mathbf{x}_t) \ge Q_{1 - \beta}(\mathcal{H}_{\mathrm{disc}})$.
  * **Hybrid Budget Allocation (`P1` - Đề xuất):** Chia cứng ngân sách cửa sổ $B$ thành hai hạn ngạch độc lập:
    * Hạn ngạch Khai thác (Top-Risk): $B_{\mathrm{risk}} = \lfloor \rho B \rfloor$ với $\rho = 0.7$, ngưỡng $Q_{1 - \rho \beta}(\mathcal{H}_{\hat{p}})$.
    * Hạn ngạch Khám phá (HST-Tree Discordance): $B_{\mathrm{disc}} = B - B_{\mathrm{risk}}$ ($30\%$), ngưỡng $Q_{1 - (1-\rho)\beta}(\mathcal{H}_{\mathrm{disc}})$.

---

## 5. Ma Trận Thực Nghiệm Đối Chứng (7 Cấu Hình)

Chạy trên cùng `seed=42`, cùng tệp dữ liệu, cùng hàng đợi nhãn trễ và cùng ngân sách $\beta = 0.005$ cho tất cả các nhánh Active Learning:

| Mã | Tên cấu hình | Ngân sách AL | Vai trò trong báo cáo |
| :--- | :--- | :---: | :--- |
| `B0_Batch` | Static XGBoost / RF (train trên Prefix cắt $D$) | $0$ | Mốc đối chứng Batch tĩnh không cập nhật |
| `B0_Passive` | Online ARF thuần (không Active Learning) | $0$ | Mốc cơ sở Online thụ động chịu trễ nhãn hoàn toàn |
| `B1_Random` | Online ARF + Random Verification | $\beta$ | Kiểm chứng sàn (chiến lược thông minh phải vượt chọn bừa) |
| `B2_TopRisk` | Online ARF + 100% Top-Risk Verification | $\beta$ | Kiểm chứng thuần Khai thác (dễ dính thiên kiến tự củng cố) |
| `B3_Margin` | Online ARF + 100% Adaptive Margin | $\beta$ | Kiểm chứng thuần Khám phá kiểu Uncertainty Sampling |
| `B4_Discord` | Online ARF + 100% HST-Tree Discordance | $\beta$ | Kiểm chứng thuần Khám phá bằng bất đồng Hybrid HST-Tree |
| `P1_Hybrid` | **Online ARF + Hybrid (70% Top-Risk + 30% Discordance)** | $\beta$ | **Phương pháp đề xuất cân bằng Khai thác và Khám phá** |

---

## 6. Bộ Chỉ Số Đánh Giá & Kiểm Thử Bắt Buộc

### 6.1. Chỉ số xuất ra `results/benchmark_summary.csv`
1. **Hiệu năng xếp hạng (độc lập ngưỡng):** Global `PR-AUC (AP)`, Chunk-Mean `PR-AUC`, `Pr@k` ($\alpha = 1.0$).
2. **Hiệu năng vận hành tại ngưỡng $\theta_c$:** `Recall`, `Precision`, `False Positives (FP)`.
3. **Hiệu quả Active Learning:** Tổng số lượt xác minh $|\mathcal{Q}|$, tỷ lệ ngân sách thực dùng $\hat{\beta}$, độ trễ nhãn trung bình của các ca gian lận (giây), và **Verification Yield**:
   $$
   \mathrm{Yield} = \frac{\sum_{i \in \mathcal{Q}} \mathbb{I}(y_i = 1)}{|\mathcal{Q}|}
   $$
4. **Chi phí hệ thống (SLA):** `Mean Latency (ms/tx)` và `p95 Latency (ms/tx)` (mục tiêu $< 5.0\text{ ms/tx}$).

### 6.2. Bộ kiểm thử tự động (`pytest tests/test_al_pipeline.py`)
Mọi lần chạy báo cáo chỉ hợp lệ khi vượt qua $4/4$ bài test sau:
1. `test_no_leakage_on_predict`: Đảm bảo `predict_proba_one` không bao giờ kích hoạt `learn_one` và mọi nhãn đều chỉ được nạp khi `current_time >= label_arrival_time`.
2. `test_no_time_feature`: Xác nhận cột `Time` không tồn tại trong từ điển đặc trưng `x` truyền vào cả `Tree` và `HST`.
3. `test_strict_budget_cap`: Xác nhận tổng số truy vấn AL trên mỗi cửa sổ không vượt quá $B$ và tổng toàn luồng không vượt quá $\lceil \beta \cdot N \rceil$.
4. `test_hst_semi_supervised_purity`: Xác nhận mô-đun `HalfSpaceTrees` chỉ tăng bộ đếm cập nhật khi $y = 0$, hoàn toàn bỏ qua khi $y = 1$.

---

## 7. Ràng Buộc Chống Lỗi Khi Vibe Coding (Guardrails)

* **Tương thích thư viện `river`:** Kiểm tra đúng đường dẫn import của `ARFClassifier` (`river.forest.ARFClassifier` trên bản mới, fallback `river.ensemble` nếu cần). Không truyền tham số `w=` trực tiếp vào `ARFClassifier.learn_one(x, y)`.
* **Chuẩn hóa bắt buộc cho `HalfSpaceTrees`:** Luôn bọc `preprocessing.MinMaxScaler()` trước `anomaly.HalfSpaceTrees()` (hoặc thiết lập `limits` rõ ràng) vì đầu vào của ULB (`V1`–`V28`, `Amount`) không nằm trong $[0, 1]$.
* **Không bịa số liệu:** Toàn bộ bảng trong báo cáo và biểu đồ phải được sinh tự động từ file `results/benchmark_summary.csv` thực chạy, kèm lệnh tái lập cụ thể trong `README.md` (`python experiments/run_benchmark.py --seed 42 --budget 0.005`).