# 🏠 Dashboard Phân tích Giá Nhà

Dashboard tương tác để phân tích và dự đoán giá nhà sử dụng Streamlit và Machine Learning.

## 📊 Tính năng chính

### 🔍 **Bộ lọc dữ liệu**

- Lọc theo khoảng giá
- Lọc theo chất lượng nhà
- Lọc theo năm xây dựng
- Lọc theo diện tích sinh hoạt
- Lọc theo khu vực
- Lọc theo số phòng tắm
- Lọc theo sức chứa xe trong gara
- Lọc theo diện tích lô đất
- Lọc theo diện tích tầng hầm
- Lọc theo diện tích gara
- Lọc theo loại nhà
- Lọc theo điều kiện bán
- Lọc theo hệ thống sưởi/làm mát

### 📈 **Phân tích dữ liệu**

- **Phân tích Tổng quan**: Histogram giá nhà, biến động giá theo năm
- **Phân tích Khu vực**: Giá nhà theo khu vực, top 5 khu vực có giá cao nhất
- **Phân tích Chất lượng**: Boxplot giá theo chất lượng, scatter plot diện tích vs giá
- **Phân tích Tương quan**: Heatmap tương quan, biểu đồ tương quan với giá nhà
- **Mô hình Dự đoán**: Linear Regression, Random Forest với dự đoán giá nhà mới

### 🎨 **Tùy chỉnh giao diện**

- Chủ đề biểu đồ (plotly, ggplot2, seaborn, v.v.)
- Hiển thị/ẩn lưới và chú thích
- Bảng màu (Viridis, Plasma, Inferno, v.v.)
- Kiểu biểu đồ (Minimal, Professional, Colorful, Dark Theme, Light Theme)

### 📋 **Tùy chọn hiển thị**

- Thống kê mô tả
- Phân tích tương quan
- Phân tích outliers
- Thông tin dữ liệu
- Tầm quan trọng thuộc tính

## 🚀 Cách chạy locally

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
streamlit run app.py
```

### 3. Truy cập dashboard

Mở trình duyệt và truy cập: `http://localhost:8501`

## 🌐 Deploy lên Streamlit Cloud

### Bước 1: Chuẩn bị repository

1. Tạo repository trên GitHub
2. Upload tất cả file lên repository:
   - `app.py`
   - `requirements.txt`
   - `house_price.csv`
   - `data_description.txt`
   - `.gitignore`
   - `README.md`

### Bước 2: Deploy lên Streamlit Cloud

1. Truy cập [share.streamlit.io](https://share.streamlit.io)
2. Đăng nhập bằng GitHub
3. Click "New app"
4. Chọn repository và branch
5. Đặt đường dẫn file chính: `app.py`
6. Click "Deploy!"

### Bước 3: Cấu hình (tùy chọn)

- **Secrets**: Nếu cần API keys, thêm vào Streamlit Cloud secrets
- **Resources**: Có thể tăng CPU/RAM nếu cần

## 📁 Cấu trúc dự án

```
Khai_pha_du_lieu/
├── app.py                 # Ứng dụng Streamlit chính
├── requirements.txt       # Dependencies
├── house_price.csv       # Dữ liệu nhà
├── data_description.txt  # Mô tả thuộc tính
├── .gitignore           # File loại trừ Git
├── README.md            # Hướng dẫn dự án
└── Phân_tích_các_yếu_tố_ảnh_hưởng_đến_giá_nhà_ở_American.ipynb  # Notebook gốc
```

## 🛠️ Công nghệ sử dụng

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Icons**: Bootstrap Icons

## 📊 Dữ liệu

Dataset: House Prices - Advanced Regression Techniques

- **Số mẫu**: 2,919 nhà
- **Thuộc tính**: 80 biến
- **Mục tiêu**: Dự đoán giá nhà (SalePrice)

## 👥 Tác giả

- Đoàn Thế Hiếu
- Nguyễn Quang Hệ
- Ngô Mạnh Minh Huy

## 📝 License

Dự án này được tạo cho mục đích học tập và nghiên cứu.

---

**Lưu ý**: Đảm bảo file `house_price.csv` có trong repository để dashboard hoạt động đúng cách.
