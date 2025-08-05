import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(
    page_title="Phân tích giá nhà - Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.markdown('<h1 class="main-header">Dashboard Phân tích Giá Nhà</h1>', unsafe_allow_html=True)

# Hàm áp dụng style cho biểu đồ
def apply_chart_style(fig, chart_theme, show_grid, show_legend, color_scheme, chart_style):
    """Áp dụng style và màu sắc cho biểu đồ"""
    
    # Áp dụng template và legend
    fig.update_layout(
        template=chart_theme,
        showlegend=show_legend
    )
    
    # Nếu hiển thị legend, thêm style cho legend
    if show_legend:
        fig.update_layout(
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='gray',
                borderwidth=1
            )
        )
    
    # Áp dụng style dựa trên chart_style
    if chart_style == "Minimal":
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='black'),
            title_font_size=16
        )
    elif chart_style == "Professional":
        fig.update_layout(
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(size=14, color='#2c3e50'),
            title_font_size=18
        )
    elif chart_style == "Colorful":
        fig.update_layout(
            plot_bgcolor='#e8f4fd',
            paper_bgcolor='white',
            font=dict(size=13, color='#34495e'),
            title_font_size=17
        )
    elif chart_style == "Dark Theme":
        fig.update_layout(
            plot_bgcolor='#2c3e50',
            paper_bgcolor='#34495e',
            font=dict(size=13, color='white'),
            title_font_color='white',
            title_font_size=17
        )
    elif chart_style == "Light Theme":
        fig.update_layout(
            plot_bgcolor='#ecf0f1',
            paper_bgcolor='white',
            font=dict(size=13, color='#2c3e50'),
            title_font_size=17
        )
    
    # Áp dụng grid dựa trên show_grid
    if show_grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    else:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    
    # Áp dụng color scheme cho các biểu đồ có color
    if hasattr(fig, 'data') and len(fig.data) > 0:
        if color_scheme != "Default":
            color_maps = {
                "Viridis": "viridis",
                "Plasma": "plasma", 
                "Inferno": "inferno",
                "Magma": "magma",
                "Blues": "blues",
                "Reds": "reds",
                "Greens": "greens",
                "Purples": "purples",
                "Oranges": "oranges"
            }
            if color_scheme in color_maps:
                for trace in fig.data:
                    if hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
                        # Áp dụng color scale cho scatter plots
                        if trace.type == 'scatter':
                            trace.marker.colorscale = color_maps[color_scheme]
    
    # ĐẢM BẢO LEGEND ĐƯỢC ÁP DỤNG CUỐI CÙNG
    fig.update_layout(showlegend=show_legend)
    
    return fig

# Tải dữ liệu
@st.cache_data
def load_data():
    """Tải và xử lý dữ liệu"""
    try:
        # Tải dữ liệu
        df = pd.read_csv("house_price.csv")
        
        # Xử lý dữ liệu thiếu
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                df[column] = df[column].fillna(df[column].median())
            else:
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column] = df[column].fillna(mode_value[0])
        
        # Label encoding cho các cột categorical
        le = LabelEncoder()
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = le.fit_transform(df[column])
        
        # Bỏ cột Id
        df = df.drop('Id', axis=1)
        
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return None

# Tải dữ liệu
df = load_data()

# Sidebar
st.sidebar.title("🎛️ Điều khiển Dashboard")
st.sidebar.markdown("---")

if df is not None:
    # Tùy chọn hiển thị
    st.sidebar.markdown("### 📊 Tùy chọn hiển thị")

    # Bộ lọc giá
    price_range = st.sidebar.slider(
        "🔍 Lọc theo khoảng giá (USD)",
        min_value=int(df['SalePrice'].min()),
        max_value=int(df['SalePrice'].max()),
        value=(int(df['SalePrice'].min()), int(df['SalePrice'].max()))
    )

    # Bộ lọc chất lượng
    quality_filter = st.sidebar.selectbox(
        "🏗️ Lọc theo chất lượng",
        ["Tất cả", "Rất Xuất Sắc (9-10)", "Xuất Sắc (8-9)", "Tốt (7-8)", "Khá (6-7)", "Trung Bình (5-6)", "Dưới Trung Bình (1-5)"]
    )

    # Bộ lọc năm xây dựng
    year_range = st.sidebar.slider(
        "🏠 Lọc theo năm xây dựng",
        min_value=int(df['YearBuilt'].min()),
        max_value=int(df['YearBuilt'].max()),
        value=(int(df['YearBuilt'].min()), int(df['YearBuilt'].max()))
    )

    # Bộ lọc diện tích
    area_range = st.sidebar.slider(
        "📐 Lọc theo diện tích sinh hoạt (sqft)",
        min_value=500,
        max_value=5000,
        value=(500, 5000)
    )

    chart_theme = st.sidebar.selectbox(
        "🎨 Chủ đề biểu đồ",
        ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "presentation", "xgridoff", "ygridoff", "gridon"]
    )

    show_grid = st.sidebar.checkbox("📊 Hiển thị lưới", value=True)
    show_legend = st.sidebar.checkbox("📋 Hiển thị chú thích", value=True)
    
    # Tùy chọn màu sắc
    color_scheme = st.sidebar.selectbox(
        "🎨 Bảng màu",
        ["Default", "Viridis", "Plasma", "Inferno", "Magma", "Blues", "Reds", "Greens", "Purples", "Oranges"]
    )
    
    # Tùy chọn style
    chart_style = st.sidebar.selectbox(
        "🎭 Kiểu biểu đồ",
        ["Standard", "Minimal", "Professional", "Colorful", "Dark Theme", "Light Theme"]
    )

    # Tùy chọn mô hình
    st.sidebar.markdown("### 🤖 Tùy chọn mô hình")

    selected_models = st.sidebar.multiselect(
        "🎯 Chọn mô hình dự đoán",
        ["Linear Regression", "Random Forest", "Polynomial Regression"],
        default=["Linear Regression", "Random Forest"]
    )

    # Tùy chọn hiển thị dữ liệu
    st.sidebar.markdown("### 📋 Tùy chọn dữ liệu")

    show_stats = st.sidebar.checkbox("📊 Hiển thị thống kê", value=True)
    show_correlation = st.sidebar.checkbox("🔗 Hiển thị tương quan", value=True)
    show_outliers = st.sidebar.checkbox("⚠️ Hiển thị outliers", value=False)

    # Thông tin dự án
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Thông tin dự án")
    st.sidebar.markdown(f"""
    - **Dữ liệu**: House Prices Dataset
    - **Số mẫu**: {len(df):,} nhà
    - **Thuộc tính**: {len(df.columns)} biến
    - **Mục tiêu**: Dự đoán giá nhà
    """)

else:
    # Giá trị mặc định khi không có dữ liệu
    price_range = (0, 1000000)
    quality_filter = "Tất cả"
    year_range = (1900, 2010)
    area_range = (500, 5000)
    chart_theme = "plotly"
    show_grid = True
    color_scheme = "Default"
    chart_style = "Standard"
    selected_models = ["Linear Regression", "Random Forest"]
    show_stats = True
    show_correlation = True
    show_outliers = False

if df is not None:
    # Áp dụng bộ lọc từ sidebar
    filtered_df = df.copy()
    
    # Lọc theo khoảng giá
    filtered_df = filtered_df[
        (filtered_df['SalePrice'] >= price_range[0]) & 
        (filtered_df['SalePrice'] <= price_range[1])
    ]
    
    # Lọc theo chất lượng
    if quality_filter != "Tất cả":
        if "Rất Xuất Sắc" in quality_filter:
            filtered_df = filtered_df[filtered_df['OverallQual'] >= 9]
        elif "Xuất Sắc" in quality_filter:
            filtered_df = filtered_df[(filtered_df['OverallQual'] >= 8) & (filtered_df['OverallQual'] < 9)]
        elif "Tốt" in quality_filter:
            filtered_df = filtered_df[(filtered_df['OverallQual'] >= 7) & (filtered_df['OverallQual'] < 8)]
        elif "Khá" in quality_filter:
            filtered_df = filtered_df[(filtered_df['OverallQual'] >= 6) & (filtered_df['OverallQual'] < 7)]
        elif "Trung Bình" in quality_filter:
            filtered_df = filtered_df[(filtered_df['OverallQual'] >= 5) & (filtered_df['OverallQual'] < 6)]
        elif "Dưới Trung Bình" in quality_filter:
            filtered_df = filtered_df[filtered_df['OverallQual'] < 5]
    
    # Lọc theo năm xây dựng
    filtered_df = filtered_df[
        (filtered_df['YearBuilt'] >= year_range[0]) & 
        (filtered_df['YearBuilt'] <= year_range[1])
    ]
    
    # Lọc theo diện tích sinh hoạt
    filtered_df = filtered_df[
        (filtered_df['GrLivArea'] >= area_range[0]) & 
        (filtered_df['GrLivArea'] <= area_range[1])
    ]
    
    # Thông tin tổng quan
    st.markdown("## 📊 Thông tin tổng quan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tổng số mẫu", f"{len(filtered_df):,}")
    
    with col2:
        st.metric("Số thuộc tính", f"{len(filtered_df.columns)}")
    
    with col3:
        avg_price = filtered_df['SalePrice'].mean()
        st.metric("Giá trung bình", f"${avg_price:,.0f}")
    
    with col4:
        max_price = filtered_df['SalePrice'].max()
        st.metric("Giá cao nhất", f"${max_price:,.0f}")
    
    # Hiển thị thông tin bộ lọc
    if len(filtered_df) != len(df):
        st.info(f"Đang hiển thị {len(filtered_df):,} mẫu từ {len(df):,} mẫu gốc")
    
    st.markdown("---")
    
    # Tabs cho các phân tích khác nhau
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Phân tích Tổng quan", 
        "🏘️ Phân tích Khu vực", 
        "🏗️ Phân tích Chất lượng", 
        "📊 Tương quan", 
        "🤖 Mô hình Dự đoán"
    ])
    
    with tab1:
        st.markdown("## 📈 Phân tích Tổng quan")
        
        # Phân bổ giá nhà
        col1, col2 = st.columns(2)
        
        with col1:
            # Tạo histogram bằng go.Figure để có legend rõ ràng
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered_df['SalePrice'],
                nbinsx=30,
                name='Phân bổ giá nhà',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title="Phân bổ giá nhà",
                xaxis_title="Giá nhà (USD)",
                yaxis_title="Số lượng",
                height=400
            )
            fig = apply_chart_style(fig, chart_theme, show_grid, show_legend, color_scheme, chart_style)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Biến động giá theo năm - Tạo bằng go.Figure để có legend rõ ràng
            yearly_avg = filtered_df.groupby('YrSold')['SalePrice'].mean().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_avg['YrSold'],
                y=yearly_avg['SalePrice'],
                mode='lines+markers',
                name='Giá trung bình',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title="Biến động giá nhà theo năm",
                xaxis_title="Năm bán",
                yaxis_title="Giá trung bình (USD)",
                height=400
            )
            fig = apply_chart_style(fig, chart_theme, show_grid, show_legend, color_scheme, chart_style)
            st.plotly_chart(fig, use_container_width=True)
        

        
        # Thống kê mô tả
        if show_stats:
            st.markdown("### 📋 Thống kê mô tả")
            numeric_cols = filtered_df.select_dtypes(include=['number'])
            st.dataframe(numeric_cols.describe(), use_container_width=True)
        
        # Hiển thị outliers nếu được chọn
        if show_outliers and len(filtered_df) > 0:
            st.markdown("### ⚠️ Phân tích Outliers")
            
            # Tính Q1, Q3 và IQR cho giá nhà
            Q1 = filtered_df['SalePrice'].quantile(0.25)
            Q3 = filtered_df['SalePrice'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Xác định outliers
            outliers = filtered_df[
                (filtered_df['SalePrice'] < Q1 - 1.5 * IQR) | 
                (filtered_df['SalePrice'] > Q3 + 1.5 * IQR)
            ]
            
            if len(outliers) > 0:
                st.warning(f"Phát hiện {len(outliers)} outliers trong dữ liệu")
                st.dataframe(outliers[['SalePrice', 'OverallQual', 'GrLivArea', 'YearBuilt']].head(), use_container_width=True)
            else:
                st.success("Không phát hiện outliers trong dữ liệu đã lọc")
        elif not show_outliers:
            st.info("⚠️ Tùy chọn 'Hiển thị outliers' đã được tắt. Vui lòng bật lại trong sidebar để xem phân tích outliers.")
    
    with tab2:
        st.markdown("## 🏘️ Phân tích Khu vực")
        
        # Mapping tên khu vực
        neighborhood_mapping = {
            0: "Bloomington Heights",
            1: "Bluestem", 
            2: "Briardale",
            3: "Brookside",
            4: "Clear Creek",
            5: "College Creek",
            6: "Crawford",
            7: "Edwards",
            8: "Gilbert",
            9: "Iowa DOT and Rail Road",
            10: "Meadow Village",
            11: "Mitchell",
            12: "North Ames",
            13: "Northridge",
            14: "Northpark Villa",
            15: "Northridge Heights",
            16: "Northwest Ames",
            17: "Old Town",
            18: "South & West of Iowa State University",
            19: "Sawyer",
            20: "Sawyer West",
            21: "Somerset",
            22: "Stone Brook",
            23: "Timberland",
            24: "Veenker"
        }
        
        # Tính giá trung bình theo khu vực
        neighborhood_prices = filtered_df.groupby('Neighborhood')['SalePrice'].mean().reset_index()
        neighborhood_prices['Neighborhood_Name'] = neighborhood_prices['Neighborhood'].map(neighborhood_mapping)
        neighborhood_prices = neighborhood_prices.sort_values('SalePrice', ascending=False)
        
        # Tạo bar chart bằng go.Figure để có legend rõ ràng
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=neighborhood_prices['Neighborhood_Name'],
            y=neighborhood_prices['SalePrice'],
            name='Giá trung bình',
            marker_color='lightgreen'
        ))
        fig.update_layout(
            title="Giá nhà trung bình theo khu vực",
            xaxis_title="Khu vực",
            yaxis_title="Giá trung bình (USD)",
            xaxis=dict(tickangle=45),
            height=500
        )
        fig = apply_chart_style(fig, chart_theme, show_grid, show_legend, color_scheme, chart_style)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 5 khu vực có giá cao nhất
        st.markdown("### 🏆 Top 5 khu vực có giá cao nhất")
        top_5 = neighborhood_prices.head()
        color_scale = 'viridis' if color_scheme == "Default" else color_scheme.lower()
        
        # Tạo bar chart với legend rõ ràng
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_5['Neighborhood_Name'],
            y=top_5['SalePrice'],
            name='Top 5 khu vực',
            marker=dict(
                color=top_5['SalePrice'],
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(title="Giá (USD)")
            )
        ))
        fig.update_layout(
            title="Top 5 khu vực có giá cao nhất",
            xaxis_title="Khu vực",
            yaxis_title="Giá (USD)",
            height=400
        )
        fig = apply_chart_style(fig, chart_theme, show_grid, show_legend, color_scheme, chart_style)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## 🏗️ Phân tích Chất lượng")
        
        # Mapping chất lượng
        qual_labels_vi = {
            10: 'Rất Xuất Sắc',
            9: 'Xuất Sắc',
            8: 'Rất Tốt',
            7: 'Tốt',
            6: 'Khá',
            5: 'Trung Bình',
            4: 'Dưới Trung Bình',
            3: 'Yếu',
            2: 'Kém',
            1: 'Rất Kém'
        }
        
        df_temp = filtered_df.copy()
        df_temp['Chất_lượng'] = df_temp['OverallQual'].map(qual_labels_vi)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Boxplot giá theo chất lượng - Tạo bằng go.Figure để có legend rõ ràng
            fig = go.Figure()
            
            # Tạo boxplot cho từng chất lượng
            for quality in df_temp['Chất_lượng'].unique():
                data = df_temp[df_temp['Chất_lượng'] == quality]['SalePrice']
                fig.add_trace(go.Box(
                    y=data,
                    name=quality,
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title="Phân phối giá theo chất lượng tổng thể",
                xaxis_title="Chất lượng",
                yaxis_title="Giá (USD)",
                height=400
            )
            fig = apply_chart_style(fig, chart_theme, show_grid, show_legend, color_scheme, chart_style)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot diện tích vs giá - Tạo bằng go.Figure để có legend rõ ràng
            color_scale = 'viridis' if color_scheme == "Default" else color_scheme.lower()
            
            # Tạo biểu đồ scatter với legend
            fig = go.Figure()
            
            # Thêm scatter plot với colorbar
            fig.add_trace(go.Scatter(
                x=filtered_df['GrLivArea'],
                y=filtered_df['SalePrice'],
                mode='markers',
                marker=dict(
                    color=filtered_df['OverallQual'],
                    colorscale=color_scale,
                    showscale=True,
                    colorbar=dict(title="Chất lượng")
                ),
                name='Nhà theo chất lượng',
                text=filtered_df['OverallQual'],
                hovertemplate='Diện tích: %{x}<br>Giá: %{y}<br>Chất lượng: %{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Mối quan hệ diện tích - giá theo chất lượng",
                xaxis_title="Diện tích sinh hoạt (sqft)",
                yaxis_title="Giá (USD)",
                height=400
            )
            fig = apply_chart_style(fig, chart_theme, show_grid, show_legend, color_scheme, chart_style)
            st.plotly_chart(fig, use_container_width=True)
        
        # Thống kê theo chất lượng
        st.markdown("### 📊 Thống kê theo chất lượng")
        quality_stats = df_temp.groupby('Chất_lượng')['SalePrice'].agg(['count', 'mean', 'std']).round(2)
        quality_stats.columns = ['Số lượng', 'Giá trung bình', 'Độ lệch chuẩn']
        st.dataframe(quality_stats, use_container_width=True)
    
    with tab4:
        st.markdown("## 📊 Phân tích Tương quan")
        
        if show_correlation:
            # Chọn các cột quan trọng
            important_cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 
                             '1stFlrSF', '2ndFlrSF', 'LotArea', 'YearBuilt', 'GarageArea']
            
            correlation_matrix = filtered_df[important_cols].corr()
            
            # Heatmap tương quan
            color_scale = 'RdBu' if color_scheme == "Default" else color_scheme.lower()
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Ma trận tương quan giữa các yếu tố chính",
                color_continuous_scale=color_scale
            )
            fig.update_layout(height=500)
            fig = apply_chart_style(fig, chart_theme, show_grid, show_legend, color_scheme, chart_style)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tương quan với giá nhà
            corr_with_price = correlation_matrix['SalePrice'].sort_values(ascending=False)
            
            # Tạo bar chart bằng go.Figure để có legend rõ ràng
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=corr_with_price.index,
                y=corr_with_price.values,
                name='Hệ số tương quan',
                marker_color='orange'
            ))
            fig.update_layout(
                title="Tương quan với giá nhà",
                xaxis_title="Yếu tố",
                yaxis_title="Hệ số tương quan",
                height=400
            )
            fig = apply_chart_style(fig, chart_theme, show_grid, show_legend, color_scheme, chart_style)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔗 Tùy chọn 'Hiển thị tương quan' đã được tắt. Vui lòng bật lại trong sidebar để xem phân tích tương quan.")
    
    with tab5:
        st.markdown("## 🤖 Mô hình Dự đoán Giá Nhà")
        
        # Kiểm tra dữ liệu có đủ để huấn luyện mô hình không
        if len(filtered_df) < 10:
            st.warning("⚠️ Dữ liệu sau khi lọc quá ít để huấn luyện mô hình. Vui lòng mở rộng bộ lọc.")
            st.info(f"Hiện tại có {len(filtered_df)} mẫu, cần ít nhất 10 mẫu để huấn luyện mô hình.")
        else:
            # Chuẩn bị dữ liệu cho mô hình
            X = filtered_df.drop('SalePrice', axis=1)
            y = filtered_df['SalePrice']
            
            # Feature selection
            k_features = min(25, len(X.columns), len(filtered_df) - 1)
            selector = SelectKBest(score_func=mutual_info_regression, k=k_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            
            # Chia dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
        
            # Huấn luyện các mô hình
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=42)
            }
            
            results = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R²': r2,
                    'predictions': y_pred
                }
        
            # Hiển thị kết quả
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Kết quả mô hình")
                
                for name, metrics in results.items():
                    st.markdown(f"**{name}**")
                    st.metric("MAE", f"${metrics['MAE']:,.0f}")
                    st.metric("RMSE", f"${metrics['RMSE']:,.0f}")
                    st.metric("R²", f"{metrics['R²']:.4f}")
                    st.markdown("---")
            
            with col2:
                st.markdown("### 🔍 So sánh dự đoán vs thực tế")
                
                # Scatter plot cho Random Forest (mô hình tốt nhất) - Tạo bằng go.Figure để có legend rõ ràng
                fig = go.Figure()
                
                # Thêm scatter plot
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=results['Random Forest']['predictions'],
                    mode='markers',
                    name='Dự đoán vs Thực tế',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=0.7
                    )
                ))
                
                # Thêm đường chéo
                min_val = min(y_test.min(), results['Random Forest']['predictions'].min())
                max_val = max(y_test.max(), results['Random Forest']['predictions'].max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Đường chéo (Lý tưởng)',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig.update_layout(
                    title="Dự đoán vs Thực tế (Random Forest)",
                    xaxis_title="Giá thực tế",
                    yaxis_title="Giá dự đoán",
                    height=400
                )
                fig = apply_chart_style(fig, chart_theme, show_grid, show_legend, color_scheme, chart_style)
                st.plotly_chart(fig, use_container_width=True)
        
            # Dự đoán giá nhà mới
            st.markdown("### 🎯 Dự đoán giá nhà mới")
            
            col1, col2 = st.columns(2)
            
            with col1:
                overall_qual = st.slider("Chất lượng tổng thể", 1, 10, 5)
                gr_liv_area = st.slider("Diện tích sinh hoạt (sqft)", 500, 4000, 1500)
                total_bsmt_sf = st.slider("Diện tích tầng hầm (sqft)", 0, 3000, 1000)
                year_built = st.slider("Năm xây dựng", 1900, 2010, 1970)
            
            with col2:
                garage_area = st.slider("Diện tích gara (sqft)", 0, 1000, 400)
                lot_area = st.slider("Diện tích lô đất (sqft)", 1000, 50000, 10000)
                full_bath = st.slider("Số phòng tắm đầy đủ", 0, 4, 2)
                garage_cars = st.slider("Sức chứa xe trong gara", 0, 4, 2)
            
            if st.button("🔮 Dự đoán giá"):
                # Tạo dữ liệu mẫu
                sample_data = np.zeros((1, X_selected.shape[1]))
                
                # Cập nhật các giá trị được chọn
                feature_names = selected_features.tolist()
                
                # Tìm index của các features quan trọng
                if 'OverallQual' in feature_names:
                    idx = feature_names.index('OverallQual')
                    sample_data[0, idx] = overall_qual
                
                if 'GrLivArea' in feature_names:
                    idx = feature_names.index('GrLivArea')
                    sample_data[0, idx] = gr_liv_area
                
                if 'TotalBsmtSF' in feature_names:
                    idx = feature_names.index('TotalBsmtSF')
                    sample_data[0, idx] = total_bsmt_sf
                
                if 'YearBuilt' in feature_names:
                    idx = feature_names.index('YearBuilt')
                    sample_data[0, idx] = year_built
                
                if 'GarageArea' in feature_names:
                    idx = feature_names.index('GarageArea')
                    sample_data[0, idx] = garage_area
                
                if 'LotArea' in feature_names:
                    idx = feature_names.index('LotArea')
                    sample_data[0, idx] = lot_area
                
                if 'FullBath' in feature_names:
                    idx = feature_names.index('FullBath')
                    sample_data[0, idx] = full_bath
                
                if 'GarageCars' in feature_names:
                    idx = feature_names.index('GarageCars')
                    sample_data[0, idx] = garage_cars
                
                # Dự đoán
                rf_prediction = models['Random Forest'].predict(sample_data)[0]
                lr_prediction = models['Linear Regression'].predict(sample_data)[0]
                
                st.success(f"🎯 **Dự đoán giá nhà:**")
                st.metric("Random Forest", f"${rf_prediction:,.0f}")
                st.metric("Linear Regression", f"${lr_prediction:,.0f}")

else:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra file house_price.csv")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Đoàn Thế Hiếu - Nguyễn Quang Hệ - Ngô Mạnh Minh Huy | Phân tích Giá Nhà</p>
    <p>Dựa trên dữ liệu House Prices - Advanced Regression Techniques</p>
</div>
""", unsafe_allow_html=True)
