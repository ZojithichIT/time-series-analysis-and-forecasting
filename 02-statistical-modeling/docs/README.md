<div align="center">

# Lab 2: Kiểm Định ADF & Chuẩn Bị Mô Hình ARIMA / SARIMA
### Kiểm định tính dừng (ADF), Sai phân (Differencing), Định danh bậc mô hình (ACF/PACF), Ước lượng ARIMA & Mở rộng SARIMA

<p>
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter" alt="Jupyter Notebook" />
  <img src="https://img.shields.io/badge/Topic-ARIMA%20%2F%20SARIMA-brightgreen?style=for-the-badge" alt="Time-Series" />
  <img src="https://img.shields.io/badge/Library-Statsmodels%20%7C%20Pandas%20%7C%20Numpy%20%7C%20Pmdarima-purple?style=for-the-badge" alt="Libraries" />
</p>

| Thuộc tính | Chi tiết |
|---|---|
| **Notebook chính** | [`scripts/main.ipynb`](../scripts/main.ipynb) |
| **Dataset 1 (Retail Sales)** | [`data/raw/a10.csv`](../data/raw/a10.csv) (Monthly Retail Sales - Trend & Seasonality) |
| **Dataset 2 (AirPassengers)** | [`data/raw/airline-passengers.csv`](../data/raw/airline-passengers.csv) (144 quan sát, Monthly, s=12) |
| **Dữ liệu đã xử lý** | [`data/processed/`](../data/processed/) |

</div>

---

## 1. Giới Thiệu Dự Án

Dự án này tập trung vào quy trình xây dựng và đánh giá các **mô hình thống kê cổ điển trong chuỗi thời gian (ARIMA & SARIMA)**:
* Nhận diện chuỗi dừng / không dừng qua đồ thị EDA và kiểm định Augmented Dickey-Fuller (ADF).
* Áp dụng kỹ thuật sai phân bậc 1 ($d=1$) và sai phân mùa vụ ($D=1, s=12$) để đưa chuỗi về tính dừng.
* Phân tích ACF và PACF để nhận diện bậc tự hồi quy ($p, P$) và trung bình trượt ($q, Q$).
* Huấn luyện và lựa chọn mô hình dựa trên tiêu chuẩn thông tin AIC / BIC.
* Chẩn đoán phần dư (Residual Diagnostics: mean $\approx 0$, White Noise check, không tự tương quan).
* Dự báo trên tập Test (không shuffle dữ liệu) và đánh giá độ chính xác qua MAE, RMSE, MAPE.
* Mở rộng nâng cao với mô hình SARIMA trên tập dữ liệu hành khách hàng không (AirPassengers).

---

## 2. Cấu Trúc Thư Mục

```text
Lab2_TS/
├── data/
│   ├── raw/
│   │   ├── a10.csv                                  # Dataset 1: Monthly Retail Sales
│   │   └── airline-passengers.csv                   # Dataset 2: Monthly Airline Passengers
│   └── processed/                                   # Dữ liệu sau xử lý/sai phân (nếu cần lưu trữ)
├── scripts/
│   └── main.ipynb                                  # Jupyter Notebook thực hành (Bài 1 -> Bài 5 & Nâng cao)
└── docs/
    ├── README.md                                   # Trang chủ tài liệu & chỉ mục
    ├── 01_eda_and_adf_test.md                      # Tài liệu: EDA, Nhận diện chuỗi dừng & Kiểm định ADF
    ├── 02_differencing_and_identification.md       # Tài liệu: Sai phân (d, D) & Phân tích ACF/PACF (p, q)
    ├── 03_arima_modeling_and_diagnostics.md        # Tài liệu: Fit ARIMA, Chẩn đoán phần dư & Đánh giá (MAE/RMSE/MAPE)
    ├── 04_sarima_advanced.md                       # Tài liệu: Mô hình hóa SARIMA nâng cao & So sánh với ARIMA
    ├── report.md                                   # Báo cáo tổng hợp hoàn chỉnh 
    └── images/                                     # Biểu đồ và hình ảnh minh họa cho báo cáo
```

---

## 3. Danh Mục Nhiệm Vụ & Tài Liệu Dự Kiến

*  **01. Khám Phá Dữ Liệu & Kiểm Định ADF (EDA & Stationarity Test)**  
  Load dữ liệu, chuyển date index, vẽ đồ thị chuỗi thời gian, kiểm tra xu hướng, mùa vụ, thực hiện và diễn giải kiểm định ADF trên chuỗi gốc ($H_0$, p-value, ADF statistic).

*  **02. Sai Phân & Nhận Diện Bậc Mô Hình (Differencing & Model Identification)**  
  Sai phân bậc 1 ($d=1$), kiểm định ADF lại để xác nhận tính dừng. Khảo sát đồ thị ACF/PACF để xác định quy luật cắt cụt/suy giảm và đề xuất các bộ tham số $(p, d, q)$.

*  **03. Xây Dựng & Đánh Giá Mô Hình ARIMA (ARIMA Modeling & Residual Diagnostics)**  
  Chia Train (80%) / Test (20%) theo thứ tự thời gian. Ước lượng mô hình, so sánh AIC/BIC. Chẩn đoán phần dư (Residuals mean $\approx 0$, White Noise, ACF phần dư). Thực hiện dự báo và tính toán MAE, RMSE, MAPE.

*  **04. Mô Hình Hóa Nâng Cao Với SARIMA (Advanced SARIMA on AirPassengers)**  
  Nhận diện tính mùa vụ $s=12$, áp dụng kết hợp sai phân thường ($d=1$) và sai phân mùa vụ ($D=1$). Nhận diện $(p, d, q) \times (P, D, Q)_{12}$, fit SARIMA, dự báo và so sánh hiệu quả với ARIMA chuẩn.

*  **Báo Cáo Tổng Hợp (report.md)**  
  Báo cáo học thuật toàn diện giải đáp đầy đủ các câu hỏi lý thuyết và thực nghiệm trong yêu cầu bài tập.
