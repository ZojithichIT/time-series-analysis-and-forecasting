# Tài Liệu Chuyên Sâu: Thiết Lập Mô Hình ARIMA, Chẩn Đoán Phần Dư & Đánh Giá Dự Báo

> **Học phần:** Phân tích Chuỗi Thời Gian (Time Series Analysis & Forecasting)  
> **Nội dung:** Bài 5 (Chia Train/Test, Fit ARIMA, Chẩn đoán phần dư, Dự báo Out-of-sample & Đánh giá MAE/RMSE/MAPE)  
> **Dữ liệu thực nghiệm:** Monthly Anti-diabetic Drug Sales in Australia (`data/raw/a10.csv`)  
> **Notebook thực hành:** [`main.ipynb`](../main.ipynb) / [`scripts/main.ipynb`](../scripts/main.ipynb)

---

## 1. Nguyên Tắc Phân Chia Dữ Liệu Chuỗi Thời Gian

### 1.1. Tại sao KHÔNG ĐƯỢC xáo trộn (No Shuffling)?
Trong học máy truyền thống với dữ liệu độc lập cùng phân phối ($i.i.d$), việc xáo trộn ngẫu nhiên (Random Shuffling) thường được áp dụng để đảm bảo phân phối giữa các tập đồng nhất. Tuy nhiên, trong chuỗi thời gian:
1. **Phụ thuộc thời gian tuần tự (Temporal Dependency):** Giá trị $Y_t$ phụ thuộc trực tiếp vào các độ trễ $Y_{t-1}, Y_{t-2}, \dots$
2. **Ngăn chặn rò rỉ dữ liệu (Data Leakage / Look-ahead Bias):** Nếu xáo trộn, mô hình sẽ sử dụng thông tin của tương lai để dự đoán quá khứ, làm mất hoàn toàn ý nghĩa thực tế của việc kiểm thử năng lực dự báo tương lai.

### 1.2. Phân chia tập dữ liệu Dataset 1 (`a10.csv`)
* **Tổng số quan sát:** $N = 204$ tháng (1991-07 đến 2008-06).
* **Tập Train (80%):** 163 tháng (1991-07 đến 2005-01).
* **Tập Test (20%):** 41 tháng (2005-02 đến 2008-06).

---

## 2. Thiết Lập & Huấn Luyện Mô Hình ARIMA

Dựa trên kết quả định danh ở Bài 4 ($d = 1$, $p \in \{1, 2\}$, $q \in \{1\}$), ta thiết lập và huấn luyện các mô hình trên tập Train ($N_{\text{train}} = 163$ tháng). Cả hai phương án chuỗi gốc (Raw) và chuỗi logarit (Option B: Log Transformation để ổn định phương sai) đều được khảo sát:

| Tên mô hình | Thang đo dữ liệu | Cấu hình $(p, d, q) \times (P, D, Q)_s$ | AIC | BIC | Log-Likelihood | Trạng thái hội tụ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\text{ARIMA}(1, 1, 1)$ (Log - Option B)** | $\ln(Y_t)$ | $(1, 1, 1)$ | **-131.05** | **-121.79** | **68.52** | Đạt hội tụ (`True`) |
| $\text{ARIMA}(0, 1, 1)$ (Log) | $\ln(Y_t)$ | $(0, 1, 1)$ | -120.45 | -114.27 | 62.22 | Đạt hội tụ (`True`) |
| **$\text{ARIMA}(1, 1, 1)$ (Raw)** | $Y_t$ | $(1, 1, 1)$ | 582.47 | 591.73 | -288.24 | Đạt hội tụ (`True`) |
| **$\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$** | $Y_t$ | $(1, 1, 1) \times (1, 1, 1)_{12}$ | **245.41** | **260.46** | **-117.70** | Đạt hội tụ (`True`) |

---

## 3. Chẩn Đoán Phần Dư (Residual Diagnostics)

Phần dư được định nghĩa là:
$$e_t = Y_t - \hat{Y}_t$$
Để một mô hình được coi là tối ưu, phần dư phải thỏa mãn các điều kiện của **Nhiễu trắng (White Noise)**:
1. Kỳ vọng toán học xấp xỉ 0: $\text{Mean}(e_t) \approx 0$ (Không chệch).
2. Phương sai không đổi: $\text{Var}(e_t) = \sigma^2 < \infty$.
3. Không tự tương quan: $\text{Cov}(e_t, e_{t-k}) = 0, \forall k \neq 0$.

![Chẩn đoán phần dư](images/05_residual_diagnostics_a10.png)

### Bảng Kiểm Định Phần Dư:
| Mô hình | Mean phần dư | Đạt $\text{Mean} \approx 0$? | Ljung-Box p-val (Lag 1) | Ljung-Box p-val (Lag 12) | Kết luận Nhiễu trắng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **$\text{ARIMA}(1, 1, 1)$ (Log - Option B)** | **0.0442** | **Thỏa mãn ($\approx 0$)** | **0.9571 ($>0.05$)** | **$5.19 \times 10^{-13}$ ($<0.05$)** | **KHÔNG đạt** (Còn mùa vụ tại lag 12) |
| $\text{ARIMA}(1, 1, 1)$ (Raw) | 0.0881 | Thỏa mãn ($\approx 0$) | 0.8842 ($>0.05$) | $2.45 \times 10^{-31}$ ($<0.05$) | **KHÔNG đạt** (Còn mùa vụ tại lag 12) |
| **$\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$** | **0.0342** | **Thỏa mãn ($\approx 0$)** | **0.9412 ($>0.05$)** | **0.4285 ($>0.05$)** | **ĐẠT NHIỄU TRẮNG HOÀN TOÀN** |

---

## 4. Dự Báo Out-of-Sample & Đánh Giá Trên Tập Test (41 Tháng)

![Đồ thị dự báo out-of-sample](images/05_forecast_evaluation_a10.png)

### Bảng Chỉ Số Đánh Giá Bắt Buộc (MAE, RMSE, MAPE):
| Mô hình | MAE (Triệu $) | RMSE (Triệu $) | MAPE (%) | Sai lệch lớn nhất (Max Error) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **$\text{ARIMA}(1, 1, 1)$ (Log - Option B)** | **3.8329** | **4.8714** | **18.49%** | **14.2882** (tháng `2008-01`) | **Trung bình (Bám tốt Trend, bỏ lỡ mùa vụ)** |
| $\text{ARIMA}(1, 1, 1)$ (Raw) | 3.6667 | 4.6243 | 18.04% | 13.7975 (tháng `2008-01`) | Trung bình (Bỏ lỡ mùa vụ) |
| **$\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$** | **1.5608** | **2.0023** | **9.06%** | **6.8412** (tháng `2008-01`) | **Rất tốt (Kiểm soát cả trend lẫn mùa vụ)** |

---

## 5. Giải Thích Ý Nghĩa Thống Kê & Bản Chất Lý Thuyết

* **Vì sao RMSE nhạy với Outlier nhất:** Do thành phần sai số $(Y_t - \hat{Y}_t)^2$ bị khuếch đại bậc hai, một sai số cực đại đơn lẻ tại tháng đỉnh điểm sẽ kéo chỉ số RMSE tăng vọt.
* **RMSE phản ánh điều gì:** Độ phân tán (Standard Error) của sai số dự báo theo cùng đơn vị đo lường thực tế.
* **Đường dự báo của ARIMA thuần túy:** Do chỉ dựa trên độ trễ 1-2 tháng phi mùa vụ, khi dự báo xa nhiều bước (Multi-step ahead forecast), quán tính suy giảm dần và đường dự báo tiệm cận về một đường nằm ngang, không nắm bắt được chu kỳ mùa hè/đông hàng năm.

---

## 6. Giải Đáp Toàn Diện 9 Câu Hỏi Bài 5

### Câu 1: Vì sao không được shuffle dữ liệu chuỗi thời gian? Test set dùng để làm gì?
* **Vì sao không được shuffle:**
  * Dữ liệu chuỗi thời gian có bản chất là **sự phụ thuộc theo thứ tự thời gian tuần tự (Temporal / Sequential Dependence)**. Giá trị quan sát ở thời điểm hiện tại $Y_t$ phụ thuộc trực tiếp vào các giá trị quá khứ liền kề ($Y_{t-1}, Y_{t-2}, \dots$).
  * Nếu thực hiện xáo trộn ngẫu nhiên (Random Shuffling) như học máy thông thường, ta sẽ phá vỡ hoàn toàn cấu trúc tự tương quan. Nghiêm trọng hơn, shuffle sẽ gây ra lỗi **Rò rỉ dữ liệu (Data Leakage / Look-ahead bias)**: Mô hình ở thời điểm quá khứ lại vô tình được "nhìn trước" các điểm dữ liệu trong tương lai, dẫn đến việc mô hình có kết quả cao giả tạo nhưng thất bại khi dự báo thực tế.
* **Test set dùng để làm gì:**
  * Tập Test đóng vai trò là "tương lai chưa biết" (Out-of-sample). Test set được dùng để **đánh giá khách quan năng lực tổng quát hóa (Generalization capability)** và độ chính xác thực tế của mô hình khi đối mặt với dữ liệu hoàn toàn mới chưa từng xuất hiện trong quá trình huấn luyện.

---

### Câu 2: Ý nghĩa của các tham số $p, d, q$ đã chọn trong mô hình $\text{ARIMA}(1, 1, 1)$?
* **$d = 1$ (Bậc sai phân tích hợp):**
  * Thể hiện chuỗi gốc $Y_t$ là chuỗi không dừng có xu hướng ngẫu nhiên (Stochastic Trend, tích hợp bậc 1: $I(1)$). Phép sai phân bậc 1 ($\Delta \ln(Y_t) = \ln(Y_t) - \ln(Y_{t-1})$) giúp triệt tiêu xu hướng tăng trưởng dài hạn, đưa chuỗi về trạng thái dừng có kỳ vọng ổn định quanh 0 ($E \approx 0$).
* **$p = 1$ (Bậc tự hồi quy AR):**
  * Đại diện cho **quán tính ngắn hạn** của thị trường: Tốc độ tăng trưởng doanh số của tháng này chịu ảnh hưởng trực tiếp từ tốc độ tăng trưởng của tháng liền kề trước đó ($Y_{t-1}$) theo tỷ lệ hệ số hồi quy $\phi_1$.
* **$q = 1$ (Bậc trung bình trượt MA):**
  * Đại diện cho **cơ chế điều chỉnh sai số / cú sốc**: Doanh số tháng này được bù trừ/điều chỉnh bởi các cú sốc ngẫu nhiên hoặc sai lệch dự báo phát sinh từ tháng trước ($\epsilon_{t-1}$) theo hệ số trọng số $\theta_1$.

---

### Câu 3: Mô hình có hội tụ không?
* **Trả lời:** **Có, mô hình $\text{ARIMA}(1, 1, 1)$ hội tụ thành công, ổn định và nhanh chóng.**
* **Minh chứng kỹ thuật:** 
  * Thuật toán ước lượng khả năng tối đa (Maximum Likelihood Estimation - MLE) sử dụng phương pháp tối ưu hóa L-BFGS hoàn thành hội tụ (`converged = True`) mà không gặp bất kỳ cảnh báo `ConvergenceWarning` nào.
  * Các tham số ước lượng nằm hoàn toàn trong miền khả nghịch (Invertibility) và dừng (Stationarity), ma trận hiệp phương sai của các tham số (Hessian matrix) xác định dương, chứng minh mô hình đạt nghiệm tối ưu toàn cục vững chắc.

---

### Câu 4: Residuals có phải white noise không?
* **Trả lời:** **Phần dư CHƯA PHẢI là Nhiễu trắng (White Noise) hoàn toàn.**
* **Minh chứng thống kê:**
  1. *Về mặt kỳ vọng (Mean):* Đạt yêu cầu lý tưởng với giá trị trung bình $\bar{e} = 0.044193 \approx 0$.
  2. *Về tính phân phối:* Biểu đồ Histogram và Q-Q plot cho thấy phần dư bám tương đối sát phân phối chuẩn ở vùng trung tâm, nhưng có đuôi dày (heavy tails) do ảnh hưởng của các tháng biến động mạnh.
  3. *Về tính độc lập (Không tự tương quan):* Lược đồ ACF của phần dư xuất hiện gai nhọn vượt dải tin cậy tại **Lag 12** và **Lag 24**. Kiểm định **Ljung–Box** tại độ trễ 12 cho thống kê $Q = 84.70$ với **p-value = $5.19 \times 10^{-13} \ll 0.05$**, bác bỏ giả thuyết $H_0$ về tính độc lập. Điều này khẳng định phần dư vẫn còn sót lại cấu trúc tự tương quan chu kỳ mùa vụ.

---

### Câu 5: Nếu residuals còn tự tương quan $\to$ Điều chỉnh gì?
* **Giải pháp điều chỉnh chuẩn tắc:**
  * Do phần dư còn tự tương quan mạnh tại các bội số của chu kỳ 12 tháng, mô hình $\text{ARIMA}$ phi mùa vụ thuần túy không đủ năng lực để mô hình hóa toàn diện.
  * Ta cần nâng cấp sang mô hình **$\text{SARIMA}(p, d, q) \times (P, D, Q)_s$** với chu kỳ mùa vụ $s = 12$:
    * Thêm phép sai phân mùa vụ bậc 1: **$D = 1$** để triệt tiêu tự tương quan chu kỳ 12 tháng ($\Delta_{12} Y_t = Y_t - Y_{t-12}$).
    * Bổ sung tham số tự hồi quy mùa vụ **$P = 1$** và/hoặc trung bình trượt mùa vụ **$Q = 1$** (ví dụ: $\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$).
  * Khi có thành phần mùa vụ, phần dư sẽ được lọc sạch hoàn toàn và trở thành White Noise chuẩn.

---

### Câu 6: Dự đoán có bám sát xu hướng không?
* **Trả lời:** **Có, đường dự báo bám sát rất tốt xu hướng tăng trưởng dài hạn (Trend) của tập Test.**
* **Phân tích:** 
  * Đường dự báo $\text{ARIMA}(1, 1, 1)$ đi lên đều đặn và nằm ngay tại trục trung tâm của tập Test thực tế (từ mức ~15 triệu $ năm 2005 lên ~21 triệu $ năm 2008).
  * Toàn bộ giá trị thực tế của tập Test nằm trọn vẹn bên trong dải tin cậy 95% (Confidence Interval) của mô hình. Tuy nhiên, đường dự báo có dạng trơn đều mà **không tái hiện được các bước nhảy vọt đỉnh/đáy mùa vụ** (Seasonal Spikes).

---

### Câu 7: Sai lệch lớn nhất xảy ra ở đâu?
* **Vị trí sai lệch cực đại:** Tập trung tại **các tháng 1 hằng năm** (tháng 01/2006, 01/2007, 01/2008) và kế tiếp là **các tháng 2 ngay sau đó**.
* **Giải thích nguyên nhân:**
  * Tại Úc, chính sách trợ cấp dược phẩm chính phủ (PBS) reset ngân sách hàng năm vào tháng 1, thúc đẩy bệnh nhân tích trữ thuốc khiến doanh số thực tế vọt đỉnh lên tới 25–29 triệu $.
  * Mô hình $\text{ARIMA}(1, 1, 1)$ chỉ dự báo mức xu hướng trung bình (~17–20 triệu $), do đó tại các tháng 1, sai số tuyệt đối tức thời bị kéo giãn cực đại lên tới **$8 - 10$ triệu $**.

---

### Câu 8: Chỉ số nào nhạy với outlier nhất?
* **Trả lời:** **$\text{RMSE}$ (Root Mean Squared Error) là chỉ số nhạy cảm nhất với outlier.**
* **Bản chất toán học:**
  * Trong công thức $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} e_t^2}$, sai số $e_t$ bị **bình phương** trước khi tính trung bình.
  * Do đó, các sai lệch cực đại (như độ lệch 10 triệu $ tại tháng 1) sẽ bị khuếch đại lên gấp bội ($10^2 = 100$), chiếm tỷ trọng chi phối trong tổng sai số. Ngược lại, $\text{MAE} = \frac{1}{n} \sum |e_t|$ chỉ đo độ lệch tuyến tính bậc 1, nên ít bị méo mó bởi các điểm đột biến hơn.
  * Thực nghiệm chứng minh: $\text{RMSE} = 4.87$ cao hơn đáng kể so với $\text{MAE} = 3.83$.

---

### Câu 9: RMSE lớn hay nhỏ nói lên điều gì?
* **Ý nghĩa thống kê:** $\text{RMSE}$ đại diện cho **độ lệch chuẩn của sai số dự báo (Sample Standard Deviation of Forecast Errors)**.
* **RMSE nhỏ:** Các sai số phân tán tập trung chặt chẽ quanh giá trị thực tế; mô hình có tính ổn định cao, không xuất hiện các dự báo trật quá xa.
* **RMSE lớn:** Cảnh báo rằng mô hình tồn tại các sai số dự báo cực đoan (Large Errors). Trong bài toán này, giá trị $\text{RMSE} = 4.87$ phản ánh trực tiếp việc mô hình bỏ lỡ các đỉnh nhọn mùa vụ của tháng 1, là minh chứng rõ ràng thúc đẩy sự cần thiết phải nâng cấp lên $\text{SARIMA}$.
