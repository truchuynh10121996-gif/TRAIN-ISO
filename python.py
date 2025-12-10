import streamlit as st
import pandas as pd
import numpy as np
import time

# Đặt cấu hình trang
st.set_page_config(
    page_title="Công cụ Tạo Dữ liệu Giao Dịch Mẫu (Fraud & Scam Detection)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chức năng chính: Tạo DataFrame ---
# Giữ cache để tăng tốc độ nếu tham số không đổi
@st.cache_data
def generate_synthetic_data(num_rows, fraud_ratio):
    """
    Tạo DataFrame chứa dữ liệu giao dịch giả lập với 24 cột đặc trưng,
    có tính toán các đặc trưng (features) và phân phối gần với thực tế,
    đặc biệt cho cả Fraud (Hack) và Scam (Lừa đảo).
    """
    st.info(f"Đang tạo {num_rows:,} dòng dữ liệu... Quá trình có thể mất vài giây.")
    
    # 1. Các cột định danh và cơ bản
    data = {}
    data['tx_id'] = np.arange(1, num_rows + 1)
    
    # Giả lập 5000 người dùng
    num_users = min(5000, num_rows // 4) 
    user_ids = np.random.choice(np.arange(100000, 100000 + num_users), num_rows)
    data['user_id'] = user_ids
    
    # Tạo thời gian giao dịch ngẫu nhiên và sắp xếp
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2024-06-01')
    timestamps = pd.to_datetime(start_date) + (end_date - start_date) * np.random.rand(num_rows)
    data['timestamp'] = timestamps
    
    # Sắp xếp theo thời gian là bắt buộc cho rolling window
    df = pd.DataFrame(data).sort_values(by='timestamp').reset_index(drop=True)

    # 2. Cột về số tiền (Amount) - Số tiền chẵn nghìn
    mu, sigma = 7, 1.5 
    amounts = np.exp(np.random.normal(mu, sigma, num_rows))
    amounts = (np.round(amounts / 1000) * 1000).astype(int)
    amounts[amounts < 1000] = 1000 
    amounts[amounts > 50000000] = (np.round(np.random.uniform(10000000, 50000000, amounts[amounts > 50000000].shape[0]) / 1000) * 1000).astype(int)
    
    df['amount'] = amounts
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_norm'] = (df['amount'] - df['amount'].min()) / (df['amount'].max() - df['amount'].min())

    # Tính percentile
    df['amount_percentile_system'] = df['amount'].rank(pct=True)

    # 3. Các cột về thời gian và cơ bản
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['account_age_days'] = np.random.randint(30, 365 * 5, num_rows)
    df['channel'] = np.random.choice([0, 1, 2], num_rows, p=[0.7, 0.2, 0.1]) # 0:MobileApp, 1:Web, 2:API
    df['location_diff_km'] = np.clip(np.random.lognormal(0.5, 1, num_rows) - 0.5, 0, 5000)
    
    # 4. Các cột hành vi (ĐÃ SỬA LỖI AttributeError)
    df['time_gap_prev_min'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0) / 60
    df['time_gap_prev_min'] = df['time_gap_prev_min'].apply(lambda x: x if x > 1 else np.random.lognormal(2, 1))

    # 1. Tạo DataFrame tạm thời với 'timestamp' làm index
    df_temp = df.set_index('timestamp')
    
    # 2. Tính velocity: group theo user_id, sử dụng .count() trên một cột bất kỳ (ví dụ 'tx_id')
    velocity_1h = df_temp.groupby('user_id')['tx_id'].rolling('1h', closed='left').count()
    velocity_24h = df_temp.groupby('user_id')['tx_id'].rolling('24h', closed='left').count()

    # 3. Gán kết quả trở lại df gốc (reset index để gán đúng vị trí)
    df['velocity_1h'] = velocity_1h.reset_index(level=0, drop=True)
    df['velocity_24h'] = velocity_24h.reset_index(level=0, drop=True)
    
    # Tiếp tục tính freq_norm
    df['freq_norm'] = (df['velocity_24h'] - df['velocity_24h'].min()) / (df['velocity_24h'].max() - df['velocity_24h'].min())
    # -----------------------------------------------------------

    df['is_new_recipient'] = np.random.choice([0, 1], num_rows, p=[0.9, 0.1]) 
    df['recipient_count_30d'] = np.clip(np.random.lognormal(1.2, 0.5, num_rows).astype(int), 1, 15)
    df['is_new_device'] = np.random.choice([0, 1], num_rows, p=[0.95, 0.05])
    df['device_count_30d'] = np.clip(np.random.lognormal(0.5, 0.3, num_rows).astype(int) + 1, 1, 5)

    # 5. Các cột MỚI để bắt Lừa đảo (Scam Features)
    
    # 5.1 amount_vs_avg_user_1m (Tỷ lệ so với trung bình 1 tháng)
    # Sửa lỗi: Cần chọn cột 'amount' để tính rolling mean
    avg_amount_1m = df_temp.groupby('user_id')['amount'].rolling('30d', closed='left').mean().reset_index(level=0, drop=True).shift(1).fillna(df['amount'].mean() * 0.5)
    df['avg_amount_1m'] = avg_amount_1m # Gán lại vào df chính
    df['amount_vs_avg_user_1m'] = df['amount'] / df['avg_amount_1m']
    df.drop(columns=['avg_amount_1m'], inplace=True)
    
    # 5.2 is_first_large_tx (Giao dịch lớn nhất từ trước đến nay)
    df['max_amount_seen'] = df.groupby('user_id')['amount'].transform(lambda x: x.cummax().shift(1).fillna(0))
    df['is_first_large_tx'] = ((df['amount'] > 2 * df['max_amount_seen']) | (df['max_amount_seen'] == 0)).astype(int)
    df.drop(columns=['max_amount_seen'], inplace=True)
    
    # 5.3 recipient_is_suspicious (Tài khoản người nhận bị đánh dấu)
    df['recipient_is_suspicious'] = np.random.choice([0, 1], num_rows, p=[0.99, 0.01])
    
    # 6. Cột gian lận (Target) và Anomaly Score
    num_fraud = int(num_rows * fraud_ratio)
    labels = np.zeros(num_rows, dtype=int)
    fraud_indices = np.random.choice(num_rows, num_fraud, replace=False)
    labels[fraud_indices] = 1
    df['is_fraud'] = labels

    # --- Điều chỉnh dữ liệu cho ATO (Hack) và SCAM (Lừa đảo) ---
    num_ato = int(num_fraud * 0.7)
    ato_indices = np.random.choice(fraud_indices, num_ato, replace=False)
    df.loc[ato_indices, 'amount'] = np.clip(np.random.lognormal(9, 1.5, num_ato), 500000, 10000000)
    df.loc[ato_indices, 'time_gap_prev_min'] = np.random.uniform(0, 10, num_ato)
    df.loc[ato_indices, 'is_new_recipient'] = 1 
    df.loc[ato_indices, 'is_new_device'] = 1 
    df.loc[ato_indices, 'location_diff_km'] = np.clip(np.random.lognormal(4, 1.5, num_ato), 50, 5000)

    scam_indices = np.setdiff1d(fraud_indices, ato_indices)
    num_scam = len(scam_indices)
    df.loc[scam_indices, 'amount'] = np.clip(np.random.lognormal(8.5, 1, num_scam), 200000, 5000000)
    df.loc[scam_indices, 'amount_vs_avg_user_1m'] = np.clip(df.loc[scam_indices, 'amount_vs_avg_user_1m'] * np.random.uniform(2, 5, num_scam), 3, 10)
    df.loc[scam_indices, 'is_first_large_tx'] = 1
    df.loc[scam_indices, 'recipient_is_suspicious'] = 1
    df.loc[scam_indices, 'is_new_device'] = 0 

    # Tính toán lại các cột phụ thuộc sau khi chỉnh sửa amount
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_norm'] = (df['amount'] - df['amount'].min()) / (df['amount'].max() - df['amount'].min())

    # Tính global_anomaly_score_prev 
    base_score = df['amount_norm'] + (1 - df['time_gap_prev_min'].clip(upper=100)/100) + df['is_new_recipient'] + df['is_new_device'] + (df['location_diff_km'].clip(upper=100)/100) + (df['amount_vs_avg_user_1m'].clip(upper=10)/10) + df['recipient_is_suspicious']
    
    df['global_anomaly_score_prev'] = (base_score + np.random.normal(0, 0.5, num_rows)) / base_score.max()
    df['global_anomaly_score_prev'] = df['global_anomaly_score_prev'].clip(0.01, 0.99)
    
    df.loc[df['is_fraud'] == 1, 'global_anomaly_score_prev'] = np.clip(df.loc[df['is_fraud'] == 1, 'global_anomaly_score_prev'] + np.random.uniform(0.1, 0.3, num_fraud), 0.7, 0.99)
    
    # 7. Chọn 24 cột đặc trưng yêu cầu và cột nhãn
    final_columns = [
        'tx_id', 'user_id', 'amount', 'amount_log', 'amount_norm', 
        'hour_of_day', 'day_of_week', 'is_weekend', 'time_gap_prev_min', 
        'velocity_1h', 'velocity_24h', 'freq_norm', 'is_new_recipient', 
        'recipient_count_30d', 'is_new_device', 'device_count_30d', 
        'location_diff_km', 'channel', 'account_age_days', 
        'amount_percentile_system', 'global_anomaly_score_prev',
        'amount_vs_avg_user_1m', 'is_first_large_tx', 'recipient_is_suspicious',
        'is_fraud'
    ]
    
    df_output = df[final_columns].copy()
    
    st.success(f"Tạo dữ liệu thành công! Tỷ lệ gian lận/lừa đảo: {df_output['is_fraud'].mean()*100:.2f}%")
    return df_output.sort_values(by='tx_id').reset_index(drop=True)


# --- Giao diện Streamlit ---
st.title("🛡️ Công cụ Tạo Dữ liệu Giao Dịch Mẫu (Fraud & Scam Detection)")

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
    "Tỷ lệ gian lận/lừa đảo mong muốn (%)",
    min_value=0.1,
    max_value=10.0,
    value=5.0,
    step=0.1,
    format="%.1f%%",
    help="Tỷ lệ mẫu bị đánh dấu là gian lận/lừa đảo (nhãn 1). Thường là 1-5% trong thực tế."
)

# Chạy nút tạo dữ liệu
if st.sidebar.button("🚀 Tạo Dữ liệu Mẫu (24 Features)"):
    start_time = time.time()
    
    # Gọi hàm tạo dữ liệu
    df_result = generate_synthetic_data(num_rows_target, fraud_ratio_target / 100)

    st.header("📊 Dữ liệu Mẫu đã Tạo")
    st.info(f"Tổng số cột đặc trưng: 24. Tổng số cột trong file CSV: 25 (bao gồm cột nhãn 'is_fraud').")
    st.dataframe(df_result.head(10)) # Hiển thị 10 dòng đầu
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tóm tắt Dữ liệu")
        st.write(df_result.describe().transpose())
        
    with col2:
        st.subheader("Phân phối Gian Lận/Lừa đảo")
        fraud_summary = df_result['is_fraud'].value_counts(normalize=True).mul(100).rename('Tỷ lệ (%)').reset_index()
        st.dataframe(fraud_summary.rename(columns={'index': 'Nhãn (0=Normal, 1=Fraud/Scam)'}))
        
        # --- Chức năng Download ---
        csv_data = df_result.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Tải file CSV mẫu về",
            data=csv_data,
            file_name=f'synthetic_fraud_scam_data_{num_rows_target}_{int(fraud_ratio_target*10)}p_fraud.csv',
            mime='text/csv',
            help="Tải file CSV chứa 24+1 cột, được thiết kế cho cả gian lận (Hack) và lừa đảo (Scam)."
        )
        
    end_time = time.time()
    st.sidebar.success(f"Hoàn thành trong {end_time - start_time:.2f} giây.")

else:
    st.info("""
    Nhấn **'🚀 Tạo Dữ liệu Mẫu (24 Features)'** ở thanh bên trái để bắt đầu tạo tập dữ liệu.

    **📝 Các Cột Đặc trưng Mới (Giúp phát hiện Lừa đảo - Scam):**
    1.  **`amount_vs_avg_user_1m`**: Tỷ lệ số tiền giao dịch hiện tại so với số tiền trung bình của người dùng trong 1 tháng trước.
    2.  **`is_first_large_tx`**: 1 nếu đây là giao dịch lớn nhất từ trước đến nay của người dùng.
    3.  **`recipient_is_suspicious`**: 1 nếu tài khoản người nhận nằm trong danh sách đen/đáng ngờ.

    **Kết quả:** Tập dữ liệu này giả lập cả giao dịch Gian lận (ATO/Hack) và Lừa đảo (Scam), mang lại sự đa dạng cao hơn cho quá trình huấn luyện mô hình.
    """)
