# BÁO CÁO THỰC HÀNH: BÀI TẬP NÂNG CAO (BTNC)
## KHÁM PHÁ & MÔ HÌNH HÓA CHUỖI THỜI GIAN AIRLINE PASSENGERS VỚI SARIMA

<div align="center">

**Học phần:** Phân Tích Chuỗi Thời Gian (Time Series Analysis & Forecasting)  
**Tập dữ liệu:** International Airline Passengers (1/1949 - 12/1960)  
**Quy mô:** 144 tháng quan sát | **Tần suất:** Hàng tháng (`Freq: MS`)  
**Tác giả thực hiện:** Dữ liệu này ban đầu được tác giả Robert G. Brown giới thiệu vào năm 1962 đo lường và thu thập tại Mỹ (Hoa Kỳ). Sau đó, nó được hai nhà thống kê George Box và Gwilym Jenkins thêm vào cuốn sách của họ. Dataset này sau đó được sử dụng rộng rãi trong các bài giảng về time series Forecasting

</div>

---

## MỤC LỤC BÁO CÁO

1. [Tổng Quan Dữ Liệu & Kiểm Tra Tính Toàn Vẹn](#1-tổng-quan-dữ-liệu--kiểm-tra-tính-toàn-vẹn)
2. [Khám Phá Chuỗi Thời Gian (EDA Chuyên Sâu)](#2-khám-phá-chuỗi-thời-gian-eda-chuyên-sâu)
   - 2.1. Đồ thị chuỗi gốc, Rolling Mean (12m) & Mốc cực trị Min/Max
   - 2.2. Phân rã chuỗi thời gian: STL (Loess) vs Mô hình Nhân (Multiplicative)
   - 2.3. Khảo sát tính Nhiễu trắng (White Noise) qua ACF của phần dư
   - 2.4. Phân tích phân bố đa chiều qua Month-wise & Year-wise Boxplot
   - 2.5. Lược đồ ACF sơ bộ nhận diện tính dừng & chu kỳ mùa vụ
3. [Quy Trình 7 Bước Mô Hình Hóa SARIMA Chuẩn Mực](#3-quy-trình-7-bước-mô-hình-hóa-sarima-chuẩn-mực)
   - Bước 1: Nhận diện chu kỳ mùa vụ ($s = 12$)
   - Bước 2: Kiểm định nghiệm đơn vị (ADF Test) trên chuỗi gốc
   - Bước 3: Sai phân kép triệt tiêu xu hướng và mùa vụ ($d=1, D=1, s=12$)
   - Bước 4: Nhận diện bậc mô hình qua ACF & PACF chuỗi sai phân kép
   - Bước 5: Phân chia tập dữ liệu Train/Test & Huấn luyện (Fit) mô hình
   - Bước 6: Dự báo ngoài mẫu (Out-of-sample Forecast $H=29$ tháng)
   - Bước 7: Bảng đối sánh trực diện & Đồ thị dự báo ngoài mẫu
4. [Kết Luận Then Chốt & Đề Xuất Thực Tiễn](#4-kết-luận-then-chốt--đề-xuất-thực-tiễn)

---

## 1. Tổng Quan Dữ Liệu & Kiểm Tra Tính Toàn Vẹn

* **Nguồn dữ liệu:** Chuỗi số lượng hành khách bay quốc tế hàng tháng từ tháng 01/1949 đến tháng 12/1960 (dữ liệu kinh điển của Box & Jenkins).
* **Kiểm tra tính liên tục:** Đặt chỉ số thời gian `DatetimeIndex` với tần suất đầu tháng (`asfreq('MS')`). Toàn bộ 144 mốc tháng liên tục, **không có giá trị khuyết thiếu (`NaN = 0`)**.
* **Thống kê mô tả tổng quan:**
  - Kích thước mẫu: $N = 144$ tháng (12 năm).
  - Giá trị trung bình ($\mu$): **280.30 nghìn khách**.
  - Độ lệch chuẩn ($\sigma$): **119.97 nghìn khách**.
  - Giá trị nhỏ nhất (MIN): **104.00 nghìn khách** (Tháng 11/1949).
  - Giá trị lớn nhất (MAX): **622.00 nghìn khách** (Tháng 07/1960).
  - Các mốc phân vị: $Q_1 (25\%) = 180.0\text{k}$, $\text{Median} (50\%) = 265.5\text{k}$, $Q_3 (75\%) = 360.5\text{k}$.

---

## 2. Khám Phá Chuỗi Thời Gian (EDA Chuyên Sâu)

### 2.1. Đồ thị chuỗi gốc, Rolling Mean (12m) & Mốc cực trị

<p align="center">
  <img src="Lab2_Timeseries/docs/images/01_ts_plot_airline.png" width="95%" alt="Đồ thị chuỗi thời gian Airline Passengers">
  <br>
  <em>Hình 1: Đồ thị chuỗi thời gian kết hợp Rolling Mean (12m), dải ±2σ và mốc cực trị Min/Max</em>
</p>

* **Đường xu hướng (Rolling Mean 12 tháng):** Dốc lên liên tục và bền vững, thể hiện sự bùng nổ của ngành hàng không dân dụng sau Chiến tranh thế giới thứ hai (tăng từ ~120k lên >500k khách, tăng gần 400% sau 12 năm).
* **Dải biến động $\pm 2\sigma$ trượt (độ lệch chuẩn):** Mở rộng rõ rệt theo thời gian. Biên độ dao động nội năm vào những năm 1949 - 1950 chỉ khoảng $\pm 30\text{k}$, nhưng đến giai đoạn 1959 - 1960 đã nới rộng lên hơn $\pm 150\text{k}$.
* **Điểm cực trị lịch sử:**
  - **Đáy lịch sử (MIN):** **104 nghìn khách** vào **tháng 11/1949** (Mùa đông thấp điểm).
  - **Đỉnh lịch sử (MAX):** **622 nghìn khách** vào **tháng 07/1960** (Mùa hè cao điểm).

---

### 2.2. Phân rã chuỗi thời gian: STL (Loess) vs Mô hình Nhân (Multiplicative)

Để bóc tách chuỗi gốc thành các thành phần cấu thành, hai phương pháp phân rã được thực hiện:

<p align="center">
  <img src="Lab2_Timeseries/docs/images/01_decomposition_airline.png" width="92%" alt="Phân rã STL Loess">
  <br>
  <em>Hình 2: Phân rã chuỗi thời gian bằng phương pháp STL (Loess, period=12, robust=True)</em>
</p>

* **Phương pháp STL (Cộng):** $Y_t = \text{Trend}_t + \text{Seasonal}_t + \text{Resid}_t$. Thuật toán Loess bóc tách đường Trend mượt mà không bị mất dữ liệu ở hai đầu. Tuy nhiên, do giả định biên độ mùa vụ không đổi qua các năm, thành phần phần dư $\text{Resid}_t$ bị phình to mạnh ở giai đoạn cuối chuỗi.

<p align="center">
  <img src="Lab2_Timeseries/docs/images/01_decomposition_multiplicative_airline.png" width="92%" alt="Phân rã Mô hình Nhân">
  <br>
  <em>Hình 3: Phân rã mô hình Nhân (Multiplicative Decomposition: Y = T × S × I)</em>
</p>

* **Phương pháp Mô hình Nhân:** $Y_t = T_t \times S_t \times I_t$. Biên độ mùa vụ được biểu diễn dưới dạng chỉ số tỷ lệ (từ 0.8 đến 1.3). Phần dư tỷ lệ $I_t$ dao động ổn định quanh trục 1.0, chứng minh tập dữ liệu mang bản chất **Mùa vụ nhân tính (Multiplicative Seasonality)**.

---

### 2.3. Khảo sát tính Nhiễu trắng (White Noise) qua ACF của phần dư

<p align="center">
  <img src="Lab2_Timeseries/docs/images/01_acf_resid_decomposition_airline.png" width="95%" alt="ACF phần dư phân rã">
  <br>
  <em>Hình 4: Đối chiếu Lược đồ ACF phần dư giữa STL (Dạng cộng) và Mô hình Nhân (Multiplicative)</em>
</p>

* **Tiêu chuẩn kiểm chứng:** Sau khi bóc tách Trend và Seasonal, phần dư lý tưởng phải là nhiễu trắng (White Noise): toàn bộ hệ số tự tương quan $\hat{\rho}_k$ phải nằm bên trong khoảng tin cậy 95% Bartlett $\left[\pm \frac{1.96}{\sqrt{N}}\right]$ và triệt tiêu tính chu kỳ mùa vụ tại các mốc $s = 12, 24, 36$.
* **Kết quả đối chiếu:** Cả hai phương pháp đều khử thành công chu kỳ mùa vụ chính tại các bội số 12. Tuy nhiên, phần dư của STL vẫn còn một số độ trễ ngắn vượt nhẹ dải tin cậy do mô hình cộng chưa thích ứng hoàn toàn với phương sai dãn nở.

---

### 2.4. Phân tích phân bố đa chiều qua Month-wise & Year-wise Boxplot

<p align="center">
  <img src="Lab2_Timeseries/docs/images/01_boxplots_airline.png" width="95%" alt="Boxplot đa chiều Airline">
  <br>
  <em>Hình 5: Month-wise Boxplot (khảo sát mùa vụ nội năm) & Year-wise Boxplot (khảo sát xu hướng và phương sai)</em>
</p>

* **Month-wise Boxplot (Bên trái):**
  - **Cao điểm mùa hè (Peak Season):** Tháng 7 và Tháng 8 có trung vị lượng khách cao nhất (>350k - 400k), phản ánh nhu cầu du lịch hè quốc tế.
  - **Thấp điểm mùa đông (Trough Season):** Tháng 11 và Tháng 2 ghi nhận mức khách thấp nhất (~200k - 230k).
* **Year-wise Boxplot (Bên phải):**
  - Đường trung vị (vạch trong hộp) tịnh tiến lên phía trên qua từng năm khẳng định xu hướng tăng trưởng bền vững.
  - Chiều cao hộp ($IQR = Q_3 - Q_1$) và độ rộng râu nến dãn rộng dần qua các năm, khẳng định hiện tượng **Phương sai thay đổi (Heteroscedasticity)**.

---

### 2.5. Lược đồ ACF sơ bộ chuỗi gốc

<p align="center">
  <img src="Lab2_Timeseries/docs/images/01_acf_airline.png" width="92%" alt="ACF chuỗi gốc">
  <br>
  <em>Hình 6: Lược đồ Tự tương quan (ACF) trên chuỗi gốc Airline Passengers</em>
</p>

* **Đặc trưng không dừng (Non-stationarity):** Hệ số ACF tại lag 1 rất cao ($\hat{\rho}_1 \approx 0.95$) và **suy giảm tuyến tính cực kỳ chậm (Slow Linear Decay)** kéo dài qua hơn 40 độ trễ.
* **Đặc trưng mùa vụ ($s=12$):** Các đỉnh sóng tự tương quan nhô cao rõ nét tại các bội số của chu kỳ năm: $\text{lag } 12, 24, 36$.
* $\implies$ **Kết luận EDA:** Chuỗi gốc vi phạm đồng thời cả tính dừng về kỳ vọng và tính thuần nhất về phương sai. Bắt buộc phải áp dụng **Sai phân kép ($d=1, D=1, s=12$)** trước khi định danh mô hình.

---

## 3. Quy Trình 7 Bước Mô Hình Hóa SARIMA Chuẩn Mực

Mô hình SARIMA có cấu trúc tổng quát $\text{SARIMA}(p, d, q) \times (P, D, Q)_s$:
$$\Phi_P(B^s) \phi_p(B) (1 - B)^d (1 - B^s)^D Y_t = \Theta_Q(B^s) \theta_q(B) \varepsilon_t$$

---

### Bước 1: Trực quan hóa & Nhận diện chu kỳ mùa vụ ($s = 12$)

<p align="center">
  <img src="Lab2_Timeseries/docs/images/06_air_eda_seasonality.png" width="95%" alt="Nhận diện mùa vụ Airline">
  <br>
  <em>Hình 7: Trực quan hóa chi tiết các chu kỳ mùa hè cao điểm (T6-T8) lặp lại hàng năm</em>
</p>

* Dữ liệu ghi nhận chu kỳ đỉnh sóng lặp lại đúng sau mỗi 12 tháng.
* Mùa hè (tháng 6 đến tháng 8 - vùng tô màu cam) luôn tạo đỉnh cục bộ của mỗi năm.

---

### Bước 2: Kiểm định nghiệm đơn vị (ADF Test) trên chuỗi gốc

Kiểm định Augmented Dickey-Fuller (ADF) được tiến hành với giả thuyết $H_0$: Chuỗi có nghiệm đơn vị (không dừng).

| Chỉ số kiểm định | Giá trị thực nghiệm | Ngưỡng 5% | Kết luận thống kê |
| :--- | :---: | :---: | :--- |
| **ADF Statistic** | **+0.8154** | -2.8840 | ADF Statistic > Ngưỡng 5% |
| **p-value** | **0.9919** | 0.05 | Không đủ cơ sở bác bỏ $H_0$ |
| **Số độ trễ (Lags)** | 13 | — | — |
| **Số quan sát (N)** | 130 | — | **Chuỗi KHÔNG DỪNG (Non-stationary)** |

---

### Bước 3: Sai phân kép triệt tiêu xu hướng và mùa vụ ($d=1, D=1, s=12$)

Tiến hành kiểm chứng tính dừng qua từng cấp độ biến đổi sai phân:
1. **Sai phân thường bậc 1 ($d=1$):** $\Delta Y_t = Y_t - Y_{t-1}$.
2. **Sai phân kép ($d=1, D=1, s=12$):** $\Delta \Delta_{12} Y_t = (1 - B)(1 - B^{12}) Y_t = (Y_t - Y_{t-1}) - (Y_{t-12} - Y_{t-13})$.

<p align="center">
  <img src="Lab2_Timeseries/docs/images/07_air_differencing_stages.png" width="95%" alt="Các giai đoạn sai phân">
  <br>
  <em>Hình 8: Biến đổi chuỗi qua 3 giai đoạn: Chuỗi gốc -> Sai phân d=1 -> Sai phân kép d=1, D=1</em>
</p>

**Bảng kết quả kiểm định ADF qua các giai đoạn sai phân:**

| Cấp độ biến đổi | ADF Statistic | p-value | Ngưỡng 5% | Trạng thái dừng |
| :--- | :---: | :---: | :---: | :--- |
| **1. Chuỗi gốc ($Y_t$)** | +0.8154 | 0.9919 | -2.8840 | **Không dừng** |
| **2. Sai phân thường ($d=1$)** | -2.8293 | 0.0542 | -2.8840 | **Chưa dừng triệt để** ($p > 0.05$, còn mùa vụ) |
| **3. Sai phân kép ($d=1, D=1, s=12$)** | **-15.5956** | **0.0000** | **-2.8840** | **DỪNG HOÀN TOÀN** ($p < 10^{-4}$) |

$\implies$ Chuỗi sai phân kép đã triệt tiêu hoàn toàn cả xu hướng tuyến tính lẫn chu kỳ mùa vụ, đủ điều kiện để nhận diện tham số.

---

### Bước 4: Nhận diện bậc mô hình qua ACF & PACF chuỗi sai phân kép

<p align="center">
  <img src="Lab2_Timeseries/docs/images/08_air_acf_pacf_d1_D1.png" width="95%" alt="ACF PACF d1 D1">
  <br>
  <em>Hình 9: Lược đồ ACF và PACF trên chuỗi sai phân kép (d=1, D=1, s=12)</em>
</p>

* **Miền phi mùa vụ (Lags 1 - 3):**
  - ACF có 1 gai âm vượt ra ngoài dải tin cậy 95% tại $\text{lag } 1$ ($\hat{\rho}_1 = -0.31$) rồi cắt cụt ngay lập tức $\implies q = 1$.
  - PACF suy giảm dần $\implies p \in \{0, 1\}$.
* **Miền mùa vụ (Lags 12, 24, 36):**
  - ACF có gai âm tại $\text{lag } 12$ ($\hat{\rho}_{12} = -0.134$), sau đó triệt tiêu ở các lag 24, 36 $\implies$ Theo mô hình Airline kinh điển xác định $Q = 1$.
  - PACF tại $\text{lag } 12$ âm nhẹ $\implies P \in \{0, 1\}$.
* **Xác định các mô hình ứng viên:** $\text{SARIMA}(p,d,q) \times (P, D, Q)_{12}$
  - Mô hình 1: $\text{SARIMA}(0, 1, 1) \times (0, 1, 1)_{12}$ (Cấu hình Airline chuẩn).
  - Mô hình 2: $\text{SARIMA}(1, 1, 1) \times (1, 1, 1)_{12}$ (Cấu hình mở rộng).
  - Mô hình đối chuẩn: $\text{ARIMA}(2, 1, 1)$ (Phi mùa vụ, để chứng minh tầm quan trọng của thành phần mùa vụ).

---

### Bước 5: Phân chia Train/Test & Huấn luyện (Fit) mô hình

* **Chiến lược phân chia:**
  - **Tập Train:** 115 tháng (từ tháng 01/1949 đến tháng 07/1958, chiếm **80%** dữ liệu).
  - **Tập Test:** 29 tháng (từ tháng 08/1958 đến tháng 12/1960, chiếm **20%** dữ liệu).

**Bảng so sánh độ khớp trong mẫu (Goodness-of-fit):**

| Mô hình | Số tham số | Log-Likelihood | AIC | BIC | Đánh giá độ khớp |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ARIMA(2, 1, 1)** | 4 | -525.32 | 1058.65 | 1069.59 | Rất kém do bỏ qua mùa vụ |
| **SARIMA(0, 1, 1)(0, 1, 1)[12]** | **3** | **-375.19** | **756.38** | **764.26** | **Tối ưu AIC và BIC (Tiết kiệm tham số)** |
| **SARIMA(1, 1, 1)(1, 1, 1)[12]** | 5 | -374.13 | 758.26 | 771.39 | AIC và BIC cao hơn do phạt thêm 2 tham số |

---

### Bước 6 & 7: Dự báo ngoài mẫu ($H=29$ tháng) & Đối sánh năng lực

Thực hiện dự báo ngoài mẫu 29 bước tương lai trên tập Test cho cả 3 mô hình và đo lường bằng các chỉ số sai số chuẩn:
$$\text{MAE} = \frac{1}{H}\sum_{t=1}^{H} |y_t - \hat{y}_t|, \quad \text{RMSE} = \sqrt{\frac{1}{H}\sum_{t=1}^{H} (y_t - \hat{y}_t)^2}, \quad \text{MAPE} = \frac{100\%}{H}\sum_{t=1}^{H} \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$

**Bảng đối sánh hiệu năng dự báo ngoài mẫu (Out-of-sample Forecast Evaluation):**

| Mô hình | AIC | BIC | MAE (Khách) | RMSE (Khách) | MAPE (%) | Đánh giá năng lực dự báo |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ARIMA(2, 1, 1)** | 1058.65 | 1069.59 | 60.53 | 82.62 | 12.90% | **Thất bại:** Đường dự báo đi ngang, mất hoàn toàn sóng đỉnh hè / đáy đông |
| **SARIMA(0, 1, 1)(0, 1, 1)[12]** | **756.38** | **764.26** | **24.08** | **30.60** | **5.16%** | **XUẤT SẮC:** Giảm **63.0% RMSE**, bám sát hoàn hảo từng chu kỳ đỉnh/đáy |
| **SARIMA(1, 1, 1)(1, 1, 1)[12]** | 758.26 | 771.39 | 23.56 | 30.14 | 5.05% | Tương đương $\text{SARIMA}_1$ nhưng cấu trúc cồng kềnh hơn |

<p align="center">
  <img src="Lab2_Timeseries/docs/images/10_air_arima_vs_sarima_forecast.png" width="95%" alt="Dự báo ARIMA vs SARIMA">
  <br>
  <em>Hình 10: Đối chiếu đường dự báo ngoài mẫu giữa ARIMA(2,1,1) và SARIMA(0,1,1)(0,1,1)[12] so với thực tế trên tập Test</em>
</p>

* **Phân tích biểu đồ dự báo:**
  - Đường dự báo của **ARIMA(2,1,1) (đường chấm đỏ)** chỉ ngoại suy một xu hướng tuyến tính phẳng xấp xỉ 450k khách, hoàn toàn bất lực trước sóng dao động mùa hè và mùa đông.
  - Đường dự báo của **SARIMA(0,1,1)(0,1,1)[12] (đường nét đứt xanh lá)** tái hiện hoàn hảo nhịp đập mùa hè tháng 7-8 đạt đỉnh trên 550k-600k khách và nhịp rơi đáy mùa đông tháng 11-2.
  - Toàn bộ các điểm dữ liệu thực tế của tập Test đều nằm trọn vẹn bên trong **Khoảng tin cậy 95% (vùng tô bóng xanh)** của mô hình SARIMA.

---

## 4. Kết Luận Then Chốt & Đề Xuất Thực Tiễn

Khi trình bày báo cáo hoặc bảo vệ trước hội đồng, 3 thông điệp học thuật cốt lõi cần nhấn mạnh:

1. **Sự vượt trội mang tính quyết định của SARIMA:**  
   Đối với chuỗi thời gian có mùa vụ rõ nét, mô hình ARIMA phi mùa vụ hoàn toàn không thể đáp ứng (sai số cao $\text{MAPE} = 12.90\%$). Việc tích hợp toán tử mùa vụ giúp giảm tới **63% RMSE** (từ 82.62 xuống 30.60 khách) và đưa sai số dự báo về mức xuất sắc **$\text{MAPE} = 5.16\%$**.
2. **Hiệu lực triệt để của Sai phân kép ($d=1, D=1, s=12$):**  
   Sai phân thường $d=1$ chỉ loại bỏ được xu hướng nhưng chưa đủ để làm chuỗi dừng ($p = 0.0542 > 0.05$). Chỉ khi kết hợp sai phân mùa vụ $D=1$ với chu kỳ $s=12$, chuỗi mới đạt trạng thái dừng hoàn toàn ($p = 0.0000$, $\text{ADF} = -15.5956$).
3. **Nguyên lý Tiết kiệm tham số (Principle of Parsimony):**  
   Mô hình kinh điển $\text{SARIMA}(0, 1, 1) \times (0, 1, 1)_{12}$ chỉ cần đúng 2 tham số trung bình trượt ($\theta_1, \Theta_1$) nhưng đạt hiệu năng xấp xỉ cấu hình phức tạp $\text{SARIMA}(1, 1, 1) \times (1, 1, 1)_{12}$, đồng thời tối ưu chỉ số $\text{AIC} = 756.38$ và $\text{BIC} = 764.26$, tránh hiện tượng quá khớp (overfitting).

