# 📈 End-to-End Time Series Analysis & Forecasting Portfolio
### Comprehensive Suite: Data Quality Diagnostics, Classical Econometrics (ARIMA/SARIMA) & Machine Learning Forecasting (XGBoost)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Econometrics-412991?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-228B22?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Time--Series%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

---

##  Giới Thiệu Tổng Quan (Executive Summary)

Dự án này là một **Portfolio chuyên sâu về Phân tích và Dự báo Chuỗi Thời Gian (Time Series Analysis & Forecasting)**, bao quát toàn bộ vòng đời phát triển giải pháp dữ liệu chuỗi thời gian: từ kiểm định chất lượng & tiền xử lý dữ liệu phức tạp (Daylight Saving Time, MCAR, Outliers), phân tích kinh tế lượng cổ điển (STL, ADF, Ljung-Box, ARIMA, SARIMA) đến mô hình học máy hiện đại (Supervised Lag Engineering, XGBoost, TimeSeriesSplit CV).

Repository được chuẩn hóa theo tiêu chuẩn công nghiệp với cấu trúc module độc lập, tài liệu kỹ thuật chi tiết (`docs/`), notebook thực nghiệm có khả năng tái lập (`scripts/main.ipynb`) và các bảng so sánh định lượng.

---

##  Năng Lực Cốt Lõi Thể Hiện Trong Dự Án (Core Competencies)

| Lĩnh vực | Kỹ thuật & Phương pháp đã triển khai |
|---|---|
| **Data Quality & Preprocessing** | Xử lý bất thường dòng thời gian thực tế: Quy ước giờ mùa hè (**Daylight Saving Time - DST** gồm Fall Back trùng lặp và Spring Forward khuyết mốc), mô phỏng và nội suy dữ liệu khuyết (**MCAR, Linear Interpolation, Forward Fill**), kiểm soát nhiễu và ngoại lai (**Z-score, IQR, Rolling Statistics**). |
| **Statistical & Econometric Analysis** | Kiểm định tính dừng (**Augmented Dickey-Fuller - ADF**), Phân rã chuỗi thời gian bằng hồi quy cục bộ (**STL Decomposition using LOESS**), phân tích hàm tự tương quan (**ACF/PACF**), kiểm định tính ngẫu nhiên (**White Noise Ljung-Box Test**). |
| **Classical Parametric Forecasting** | Biến đổi Box-Cox/Logarit để ổn định phương sai, sai phân bậc 1 (d = 1) và sai phân mùa vụ (D = 1, s = 12), xác định và tối ưu tham số mô hình **ARIMA(p, d, q)** & **SARIMA(p, d, q)(P, D, Q)s** qua **AIC / BIC / Log-Likelihood**, chẩn đoán phần dư chuẩn hóa. |
| **Machine Learning for Time Series** | Chuyển đổi chuỗi đơn biến sang bài toán học có giám sát (**Supervised Learning**), kỹ thuật trích xuất đặc trưng trễ (**Lag Features $t-1 \dots t-24$**), thống kê trượt (**Rolling Mean/Std**), phân chia **Time-based Train/Val/Test** ngăn chặn triệt để **Look-ahead Bias / Data Leakage**, huấn luyện **XGBoost Regressor / Random Forest**, tối ưu hóa với **TimeSeriesSplit**. |
| **Model Evaluation & Diagnostics** | Đo lường hiệu năng dự báo ngoài tập mẫu (Out-of-sample) với **MAE, RMSE, MAPE**, kiểm định tính độc lập và phân phối chuẩn của phần dư. |

---

##  Kiến Trúc Repository & Các Module Thực Nghiệm

```text
Timeseries-Total/
├── 📁 01-diagnostics-data-quality/       # MODULE 1: EDA, Tiền xử lý & Kiểm định chất lượng chuỗi
│   ├── data/raw & processed/            # Melbourne Daily Minimum Temperatures (3.650 ngày, 10 năm)
│   ├── docs/                            # Tài liệu lý thuyết chuyên sâu: Missing values, Outliers, STL, White Noise
│   └── scripts/main.ipynb               # Thực nghiệm EDA, mô phỏng MCAR 1%, ACF/PACF, STL
│
├── 📁 02-statistical-modeling/           # MODULE 2: Mô hình hóa thống kê cổ điển (ARIMA / SARIMA)
│   ├── data/raw & processed/            # Australian Drug Sales (a10) & Box-Jenkins AirPassengers
│   ├── docs/                            # Tài liệu phân tích: ADF test, Differencing, ARIMA vs SARIMA
│   └── scripts/main.ipynb               # Fit ARIMA/SARIMA, AIC/BIC tuning, Residual Diagnostics, Backtesting
│
└── 📁 03-advanced-forecasting/           # MODULE 3: Dự báo chuỗi thời gian với Machine Learning
    ├── data/raw & processed/            # PJM Hourly Energy Consumption - AEP (121.296 mốc giờ, ~14 năm)
    ├── docs/                            # Tài liệu kỹ thuật: DST handling, Lag feature matrix, Time split
    └── scripts/main.ipynb               # Supervised transformation (24 lags), XGBoost, TimeSeriesSplit CV
```

---

## Chi Tiết Thực Nghiệm & Kết Quả Nổi Bật

### 🔹 Module 1: Time Series Diagnostics & Data Quality (`01-diagnostics-data-quality`)
* **Tập dữ liệu:** Daily Minimum Temperatures in Melbourne (10 năm: 1981–1990, 3.650 quan sát).
* **Bài toán:** Khảo sát cấu trúc tự tương quan, kiểm định nhiễu trắng và phân tách cấu trúc thời gian.
* **Kết quả & Phát hiện then chốt:**
  * **Missing Values:** Thực nghiệm chứng minh phương pháp **Linear Interpolation** bảo toàn liên tục tốt hơn **Forward-fill** đối với chuỗi nhiệt độ có tính biến thiên mượt mà.
  * **Outliers:** Phát hiện các điểm dị biệt thời tiết qua Z-score ($|Z| > 3$) và IQR; phân tích lý thuyết khẳng định cần bảo lưu các ngoại lai thực (nhiệt độ cực đoan) vì chúng phản ánh quy luật vật lý thời tiết tự nhiên thay vì lỗi đo đạc.
  * **STL Decomposition:** Tách thành công chuỗi nhiệt độ thành Trend dài hạn, Mùa vụ chu kỳ năm ($s = 365$) và Thành phần dư (Residuals).
  * **White Noise Testing:** Kiểm định Ljung-Box xác nhận chuỗi gốc có cấu trúc tự tương quan chặt chẽ ($p \approx 0 < 0.05$), phản bác giả thuyết nhiễu trắng và mở đường cho mô hình hóa tự hồi quy AR.

---

### 🔹 Module 2: Classical Statistical Modeling – ARIMA & SARIMA (`02-statistical-modeling`)
* **Tập dữ liệu:** 
  1. Monthly Anti-diabetic Drug Sales in Australia (`a10.csv`, 204 tháng).
  2. International Airline Passengers (`airline-passengers.csv`, 144 tháng).
* **Bài toán:** Đưa chuỗi không dừng về dừng, định danh bậc mô hình Box-Jenkins và dự báo ngoài mẫu (Out-of-sample).
* **So sánh định lượng trên tập Test (41 tháng ngoài mẫu - Drug Sales):**

| Cấu hình mô hình | AIC | BIC | Phần dư đạt White Noise? | MAE (Triệu $) | RMSE (Triệu $) | MAPE (%) | Đánh giá hiệu năng |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$\text{ARIMA}(1, 1, 1)$ (Raw)** | 582.47 | 591.73 | ❌ Trượt Ljung-Box lag 12 | 3.6667 | 4.6243 | 18.04% | Bắt được trend, bỏ lỡ đỉnh mùa vụ |
| **$\text{ARIMA}(1, 1, 1)$ (Log)** | -131.05 | -121.79 | ❌ Trượt Ljung-Box lag 12 | 3.8329 | 4.8714 | 18.49% | Ổn định phương sai, chưa giải quyết tính mùa |
| **$\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$** | **245.41** | **260.46** |  **Đạt White Noise ($p=0.43 > 0.05$)** | **1.5608** | **2.0023** | **9.06%** | **Vượt trội: Giảm 50% sai số MAPE, kiểm soát toàn diện Trend & Mùa vụ** |

* **Quy trình chuẩn hóa:** 
  * Áp dụng kiểm định ADF: Chuỗi gốc $p = 1.000$ (không dừng) $\rightarrow$ Sau sai phân $d=1$: $p = 0.0003 < 0.05$ (đạt tính dừng $I(0)$).
  * Chẩn đoán phần dư khắt khe: SARIMA loại bỏ hoàn toàn tự tương quan còn sót lại tại các bước trễ mùa vụ ($s=12$).

---

### 🔹 Module 3: Machine Learning & Supervised Forecasting (`03-advanced-forecasting`)
* **Tập dữ liệu:** American Electric Power (`AEP_hourly.csv`, phụ tải điện lưới PJM Bắc Mỹ).
* **Quy mô:** **121.296 quan sát hàng giờ** liên tục qua ~14 năm (2004–2018).
* **Quy trình kỹ thuật:**
  * **Xử lý bất thường thời gian (Time-continuity pipeline):** Phát hiện và giải quyết 2 hiện tượng DST: xử lý trùng lặp giờ (Fall Back lùi 1h) bằng phép trung bình hóa, bù đắp 27 giờ khuyết thiếu (Spring Forward nhảy 1h) bằng lưới tần suất đều đặn (`asfreq('h')`) và nội suy thời gian. Đạt **100% tính liên tục** không còn mốc thời gian bị gián đoạn.
  * **Feature Engineering:** Trích xuất ma trận 24 bậc trễ ($y_{t-1}, y_{t-2}, \dots, y_{t-24}$) đại diện cho chu kỳ sinh hoạt trong ngày cùng các biến thống kê trượt (Rolling Mean 24h, Rolling Std 24h).
  * **Time-based Splitting:** Phân bổ **75% Train (90.954 mẫu) - 12.5% Val (15.159 mẫu) - 12.5% Test (15.159 mẫu)** theo đúng dòng thời gian, bảo vệ mô hình khỏi lỗi rò rỉ dữ liệu tương lai (Data Leakage).
  * **Mô hình hóa:** Huấn luyện mô hình Gradient Boosting (`XGBRegressor`) kết hợp phân tích Feature Importance để chứng minh tầm ảnh hưởng của các độ trễ chiến lược ($t-1$, $t-24$).
  
Đang trong giai đoạn hoàn thiện.

---
