import streamlit as st
import pandas as pd
import numpy as np
import time

# Đặt cấu hình trang
st.set_page_config(
    page_title="Công cụ Tạo Dữ liệu Giao Dịch Mẫu (Fraud Detection)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chức năng chính: Tạo DataFrame ---
@st.cache_data
def generate_synthetic_data(num_rows, fraud_ratio):
    """
    Tạo DataFrame chứa dữ liệu giao dịch giả lập với 21 cột,
    có tính toán các đặc trưng (features) và phân phối gần với thực tế.
    """
    st.info(f"Đang tạo {num_rows:,} dòng dữ liệu... Quá trình có thể mất vài giây.")
    
    # 1. Các cột định danh và cơ bản
    data = {}
    data['tx_id'] = np.arange(1, num_rows + 1)
    
    # Giả lập 5000 người dùng
    num_users = min(5000, num_rows // 4) 
    data['user_id'] = np.random.choice(np.arange(100000, 100000 + num_users), num_rows)
    
    # 2. Cột về số tiền (Amount) - Số tiền chẵn nghìn
    # Phân phối log-normal: hầu hết giao dịch nhỏ, một số lớn
    mu, sigma = 7, 1.5 
    amounts = np.exp(np.random.normal(mu, sigma, num_rows))
    # Làm tròn đến chẵn nghìn: /1000 -> round -> *1000
    amounts = (np.round(amounts / 1000) * 1000).astype(int)
    # Giới hạn min/max
    amounts[amounts < 1000] = 1000 
    amounts[amounts > 50000000] = (np.round(np.random.uniform(10000000, 50000000, amounts[amounts > 50000000].shape[0]) / 1000) * 1000).astype(int)
    
    data['amount'] = amounts
    data['amount_log'] = np.log1p(data['amount'])
    data['amount_norm'] = (data['amount'] - data['amount'].min()) / (data['amount'].max() - data['amount'].min())

    # Tính percentile
    data['amount_percentile_system'] = pd.Series(data['amount']).rank(pct=True).values

    # 3. Các cột về thời gian
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2024-06-01')
    
    # Tạo thời gian giao dịch ngẫu nhiên và sắp xếp
    timestamps = pd.to_datetime(start_date) + (end_date - start_date) * np.random.rand(num_rows)
    timestamps = timestamps.sort_values().reset_index(drop=True)
    
    data['timestamp'] = timestamps
    data['hour_of_day'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek # Thứ Hai=0, CN=6
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # 4. Các cột về hành vi giao dịch
    df = pd.DataFrame(data)
    df = df.sort_values(by='timestamp').reset_index(drop=True) # Sắp xếp lại theo thời gian

    # Tính time_gap_prev_min (khoảng cách thời gian với giao dịch trước)
    df['time_gap_prev_min'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0) / 60
    # Giả lập hành vi thực tế: hầu hết giao dịch cách nhau > 1 phút (thực tế)
    df['time_gap_prev_min'] = df['time_gap_prev_min'].apply(lambda x: x if x > 1 else np.random.lognormal(2, 1))

    # Tính velocity (tần suất trong 1h và 24h)
    df['velocity_1h'] = df.groupby('user_id')['timestamp'].rolling('1h', on='timestamp', closed='left').count().reset_index(level=0, drop=True)
    df['velocity_24h'] = df.groupby('user_id')['timestamp'].rolling('24h', on='timestamp', closed='left').count().reset_index(level=0, drop=True)
    df['freq_norm'] = (df['velocity_24h'] - df['velocity_24h'].min()) / (df['velocity_24h'].max() - df['velocity_24h'].min())

    # 5. Các cột về người nhận và thiết bị
    # is_new_recipient: 90% giao dịch cho người cũ
    df['is_new_recipient'] = np.random.choice([0, 1], num_rows, p=[0.9, 0.1]) 
    
    # recipient_count_30d: Số lượng người nhận khác nhau trong 30 ngày. 
    # Giả định phân phối: hầu hết người dùng giao dịch với 1-3 người
    df['recipient_count_30d'] = np.clip(np.random.lognormal(1.2, 0.5, num_rows).astype(int), 1, 15)

    # is_new_device: 95% giao dịch từ thiết bị cũ
    df['is_new_device'] = np.random.choice([0, 1], num_rows, p=[0.95, 0.05])
    
    # device_count_30d: Số lượng thiết bị khác nhau trong 30 ngày
    df['device_count_30d'] = np.clip(np.random.lognormal(0.5, 0.3, num_rows).astype(int) + 1, 1, 5)

    # 6. Các cột khác
    # location_diff_km: Phân phối nghiêng về 0 (hầu hết giao dịch tại 1 địa điểm)
    df['location_diff_km'] = np.random.lognormal(0.5, 1, num_rows)
    df['location_diff_km'] = np.clip(df['location_diff_km'] - 0.5, 0, 5000) # Chuẩn hóa lại min=0
    
    # channel: Giả lập 3 kênh: Mobile App (70%), Web (20%), API/Other (10%)
    df['channel'] = np.random.choice(['MobileApp', 'Web', 'API'], num_rows, p=[0.7, 0.2, 0.1])
    
    # account_age_days: Giả lập tài khoản có tuổi từ 30 ngày đến 5 năm
    df['account_age_days'] = np.random.randint(30, 365 * 5, num_rows)
    
    # 7. Cột gian lận (Target) và Anomaly Score
    # Tạo nhãn gian lận/bình thường (0: bình thường, 1: gian lận)
    num_fraud = int(num_rows * fraud_ratio)
    labels = np.zeros(num_rows, dtype=int)
    fraud_indices = np.random.choice(num_rows, num_fraud, replace=False)
    labels[fraud_indices] = 1
    df['is_fraud'] = labels

    # Thêm nhiễu vào dữ liệu gian lận để làm cho chúng "bất thường" (Anomaly)
    # Các giao dịch gian lận thường có:
    # - Amount lớn/rất nhỏ, Amount_percentile_system cao/thấp (hiếm gặp)
    # - Time_gap_prev_min rất nhỏ (tấn công dồn dập)
    # - Is_new_recipient=1, Is_new_device=1 (thực hiện từ tài khoản/thiết bị lạ)
    # - Location_diff_km lớn (thực hiện từ xa)
    df.loc[df['is_fraud'] == 1, 'amount'] = np.clip(np.random.lognormal(9, 1.5, num_fraud), 500000, 10000000) # Amount lớn
    df.loc[df['is_fraud'] == 1, 'time_gap_prev_min'] = np.random.uniform(0, 10, num_fraud) # Thời gian ngắn
    df.loc[df['is_fraud'] == 1, 'is_new_recipient'] = 1 
    df.loc[df['is_fraud'] == 1, 'is_new_device'] = 1
    df.loc[df['is_fraud'] == 1, 'location_diff_km'] = np.clip(np.random.lognormal(4, 1.5, num_fraud), 50, 5000) # Khoảng cách xa

    # Tính toán lại các cột phụ thuộc (log, norm) cho các giao dịch gian lận đã bị chỉnh sửa
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_norm'] = (df['amount'] - df['amount'].min()) / (df['amount'].max() - df['amount'].min())

    # global_anomaly_score_prev (Điểm bất thường toàn cầu trước đó)
    # Phân phối: hầu hết gần 0 (bình thường), outlier/fraud cao
    # Ta giả lập bằng cách cộng các đặc trưng bất thường lại
    base_score = df['amount_norm'] + (1 - df['time_gap_prev_min'].clip(upper=100)/100) + df['is_new_recipient'] + df['is_new_device'] + (df['location_diff_km'].clip(upper=100)/100)
    
    # Thêm nhiễu ngẫu nhiên và chuẩn hóa
    df['global_anomaly_score_prev'] = (base_score + np.random.normal(0, 0.5, num_rows)) / base_score.max()
    df['global_anomaly_score_prev'] = df['global_anomaly_score_prev'].clip(0.01, 0.99) # Giới hạn 0.01 - 0.99
    
    # Đảm bảo các giao dịch gian lận có điểm cao hơn
    df.loc[df['is_fraud'] == 1, 'global_anomaly_score_prev'] = np.clip(df['global_anomaly_score_prev'] + np.random.uniform(0.1, 0.3, num_fraud), 0.7, 0.99)

    # 8. Chọn 21 cột yêu cầu và bỏ cột timestamp/is_fraud (nếu muốn)
    final_columns = [
        'tx_id', 'user_id', 'amount', 'amount_log', 'amount_norm', 
        'hour_of_day', 'day_of_week', 'is_weekend', 'time_gap_prev_min', 
        'velocity_1h', 'velocity_24h', 'freq_norm', 'is_new_recipient', 
        'recipient_count_30d', 'is_new_device', 'device_count_30d', 
        'location_diff_km', 'channel', 'account_age_days', 
        'amount_percentile_system', 'global_anomaly_score_prev'
    ]

    # Thêm cột 'is_fraud' vào cuối để tiện cho việc huấn luyện (Isolation Forest không cần, nhưng LightGBM cần)
    df_output = df[final_columns + ['is_fraud']]
    
    # Chuyển đổi cột Channel sang dạng số (One-Hot Encoding)
    # Streamlit sẽ hiển thị text, nhưng cho mô hình nên dùng số
    channel_mapping = {'MobileApp': 0, 'Web': 1, 'API': 2}
    df_output['channel_code'] = df_output['channel'].map(channel_mapping)
    df_output = df_output.drop(columns=['channel'])
    df_output = df_output.rename(columns={'channel_code': 'channel'})
    
    st.success(f"Tạo dữ liệu thành công! Tỷ lệ gian lận: {df_output['is_fraud'].mean()*100:.2f}%")
    return df_output.sort_values(by='tx_id').reset_index(drop=True)


# --- Giao diện Streamlit ---
st.title("🛡️ Công cụ Tạo Dữ liệu Giao Dịch Gian Lận Mẫu (Synthetic Fraud Data)")

st.sidebar.header("⚙️ Cấu hình Dữ liệu Mẫu")
num_rows_target = st.sidebar.number_input(
    "Số lượng dòng giao dịch (rows)",
    min_value=1000,
    max_value=200000,
    value=20000,
    step=1000,
    help="Số lượng giao dịch bạn muốn tạo. Khuyến nghị 20000 dòng để train ban đầu."
)

fraud_ratio_target = st.sidebar.slider(
    "Tỷ lệ gian lận mong muốn (%)",
    min_value=0.1,
    max_value=10.0,
    value=5.0,
    step=0.1,
    format="%.1f%%",
    help="Tỷ lệ mẫu gian lận/lừa đảo (nhãn 1). Isolation Forest thường hoạt động tốt với tỷ lệ nhỏ (khoảng 1-5%)."
)

# Chạy nút tạo dữ liệu
if st.sidebar.button("🚀 Tạo Dữ liệu Mẫu"):
    start_time = time.time()
    
    # Gọi hàm tạo dữ liệu
    df_result = generate_synthetic_data(num_rows_target, fraud_ratio_target / 100)

    st.header("📊 Dữ liệu Mẫu đã Tạo")
    st.dataframe(df_result.head(10)) # Hiển thị 10 dòng đầu
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tóm tắt Dữ liệu")
        st.write(df_result.describe().transpose())
        
    with col2:
        st.subheader("Phân phối Gian Lận")
        fraud_summary = df_result['is_fraud'].value_counts(normalize=True).mul(100).rename('Tỷ lệ (%)').reset_index()
        st.dataframe(fraud_summary.rename(columns={'index': 'Nhãn (0=Normal, 1=Fraud)'}))
        
        # --- Chức năng Download ---
        csv_data = df_result.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Tải file CSV mẫu về",
            data=csv_data,
            file_name=f'synthetic_fraud_data_{num_rows_target}_{int(fraud_ratio_target*10)}p_fraud.csv',
            mime='text/csv',
            help="Tải file CSV chứa 21+1 cột (cột is_fraud được thêm vào cuối)."
        )
        
    end_time = time.time()
    st.sidebar.success(f"Hoàn thành trong {end_time - start_time:.2f} giây.")

else:
    st.info("""
    Nhấn **'🚀 Tạo Dữ liệu Mẫu'** ở thanh bên trái để bắt đầu tạo tập dữ liệu.

    **📝 Mô tả Dữ liệu:**
    * **Mục đích:** Tập dữ liệu này giả lập hành vi giao dịch thực tế, với ~5% mẫu gian lận/bất thường, lý tưởng để huấn luyện:
        * **Isolation Forest:** Dùng 21 cột đặc trưng (không dùng `is_fraud`) để tìm **outlier/anomaly** (mẫu bất thường).
        * **LightGBM/XGBoost:** Dùng 21 cột đặc trưng và cột **`is_fraud`** làm nhãn để huấn luyện mô hình phân loại (classification).
    * **Tính thực tế (85-95%):**
        * Số tiền (`amount`) phân phối log-normal, được làm tròn chẵn nghìn.
        * Khoảng cách thời gian (`time_gap_prev_min`) được giả lập > 1 phút cho hành vi bình thường.
        * `is_new_recipient/is_new_device` có tỷ lệ thấp (hầu hết là giao dịch lặp lại).
        * Các mẫu gian lận được "chỉnh sửa" để có giá trị đặc trưng bất thường (amount lớn, time_gap_prev_min nhỏ, location_diff_km lớn, v.v.).
    """)

# Thêm một phần hướng dẫn nhỏ
st.markdown("""
---
### 💡 Hướng dẫn cho Chuyên gia lập trình Python
Là một chuyên gia đã quen thuộc với Streamlit, bạn có thể dễ dàng mở rộng ứng dụng này:
1.  **Phân tích EDA:** Thêm các biểu đồ phân phối (`st.pyplot`, `st.plotly_chart`) cho `amount`, `time_gap_prev_min`, và `global_anomaly_score_prev` để trực quan hóa sự khác biệt giữa hai nhóm (`is_fraud`=0 và `is_fraud`=1).
2.  **Tùy biến cao cấp:** Thêm các tham số cho việc điều chỉnh phân phối (`mu`, `sigma` cho `amount`, `lognormal` parameters cho `location_diff_km`) vào sidebar.
""")

#
