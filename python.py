import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Đặt cấu hình trang
st.set_page_config(
    page_title="Công cụ Tạo Dữ liệu Giao Dịch Mẫu - Vietnam Fraud & Scam Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CẤU HÌNH DỮ LIỆU THỰC TẾ VIỆT NAM
# ============================================

# Mệnh giá tiền VN phổ biến để làm tròn
VN_DENOMINATIONS = [10000, 20000, 50000, 100000, 200000, 500000,
                    1000000, 2000000, 5000000, 10000000, 20000000, 50000000]

# Cấu hình amount theo loại giao dịch (min, max, weight)
TRANSACTION_AMOUNT_CONFIG = {
    'micro': {  # Cà phê, ăn vặt, grab
        'min': 15000, 'max': 200000, 'weight': 0.30,
        'common_amounts': [25000, 35000, 45000, 50000, 55000, 65000, 75000, 100000, 150000]
    },
    'small': {  # Mua sắm nhỏ, bill hàng tháng
        'min': 200000, 'max': 2000000, 'weight': 0.35,
        'common_amounts': [200000, 300000, 350000, 500000, 700000, 800000, 1000000, 1200000, 1500000]
    },
    'medium': {  # Chuyển tiền, mua đồ giá trị
        'min': 2000000, 'max': 20000000, 'weight': 0.25,
        'common_amounts': [2000000, 3000000, 5000000, 7000000, 10000000, 15000000]
    },
    'large': {  # Đầu tư, mua xe, góp nhà
        'min': 20000000, 'max': 200000000, 'weight': 0.10,
        'common_amounts': [20000000, 30000000, 50000000, 70000000, 100000000]
    }
}

# Loại giao dịch
TRANSACTION_TYPES = {
    0: 'bill_payment',      # Điện, nước, internet, điện thoại
    1: 'transfer',          # Chuyển tiền nội bộ/liên ngân hàng
    2: 'shopping',          # Mua sắm offline
    3: 'withdrawal',        # Rút tiền ATM
    4: 'ecommerce',         # Mua hàng online (Shopee, Lazada, Tiki)
    5: 'food_delivery',     # Grab, ShopeeFood, Baemin
    6: 'utilities',         # Nạp điện thoại, mua vé
    7: 'investment'         # Chứng khoán, crypto, tiết kiệm
}

# Phân bố loại giao dịch theo giờ trong ngày (VN patterns)
HOURLY_TX_TYPE_WEIGHTS = {
    # Giờ sáng sớm (5-8h): chủ yếu coffee, đi làm
    'early_morning': {'hours': [5, 6, 7], 'weights': [0.05, 0.15, 0.30, 0.10, 0.25, 0.10, 0.05, 0.0]},
    # Giờ hành chính sáng (8-12h): bill payment, transfer, shopping
    'morning_work': {'hours': [8, 9, 10, 11], 'weights': [0.20, 0.25, 0.15, 0.05, 0.15, 0.10, 0.05, 0.05]},
    # Giờ trưa (12-14h): food, shopping
    'lunch': {'hours': [12, 13], 'weights': [0.05, 0.10, 0.20, 0.05, 0.20, 0.30, 0.05, 0.05]},
    # Giờ chiều (14-18h): work transactions
    'afternoon_work': {'hours': [14, 15, 16, 17], 'weights': [0.15, 0.25, 0.20, 0.05, 0.15, 0.10, 0.05, 0.05]},
    # Giờ tối (18-22h): shopping online, food delivery
    'evening': {'hours': [18, 19, 20, 21], 'weights': [0.05, 0.10, 0.15, 0.05, 0.30, 0.25, 0.05, 0.05]},
    # Đêm khuya (22-5h): ít giao dịch, suspicious
    'night': {'hours': [22, 23, 0, 1, 2, 3, 4], 'weights': [0.05, 0.20, 0.10, 0.15, 0.30, 0.10, 0.05, 0.05]}
}

# Các loại scam phổ biến tại VN
SCAM_PATTERNS = {
    'impersonation': {  # "Công an/Viện kiểm sát gọi"
        'weight': 0.35,
        'amount_range': (10000000, 100000000),  # 10-100 triệu
        'preferred_hours': [10, 11, 14, 15, 16, 20, 21, 22],  # Giờ hành chính + tối
        'velocity_pattern': 'single_large',  # 1 giao dịch lớn đột ngột
        'device_new': False,  # Dùng device cũ (nạn nhân tự chuyển)
        'location_change': False
    },
    'job_scam': {  # "Việc làm online, đầu tư sinh lời"
        'weight': 0.25,
        'amount_range': (100000, 5000000),
        'preferred_hours': [9, 10, 11, 14, 15, 19, 20, 21],
        'velocity_pattern': 'escalating',  # Nhiều GD nhỏ → 1 GD lớn
        'device_new': False,
        'location_change': False
    },
    'lottery_prize': {  # "Trúng thưởng, nhận quà"
        'weight': 0.15,
        'amount_range': (500000, 3000000),
        'preferred_hours': [9, 10, 11, 14, 15, 16],
        'velocity_pattern': 'multiple_small',  # Nhiều GD phí nhỏ
        'device_new': False,
        'location_change': False
    },
    'romance_scam': {  # Lừa tình cảm quen qua mạng
        'weight': 0.10,
        'amount_range': (5000000, 50000000),
        'preferred_hours': [20, 21, 22, 23],  # Tối/đêm
        'velocity_pattern': 'recurring',  # GD lớn lặp lại nhiều ngày
        'device_new': False,
        'location_change': False
    },
    'investment_scam': {  # Đầu tư crypto, forex lừa đảo
        'weight': 0.15,
        'amount_range': (2000000, 100000000),
        'preferred_hours': [8, 9, 10, 14, 15, 20, 21],
        'velocity_pattern': 'increasing',  # Amount tăng dần
        'device_new': False,
        'location_change': False
    }
}

# ATO (Account Takeover) patterns - Hacker
ATO_PATTERNS = {
    'credential_theft': {  # Đánh cắp mật khẩu
        'weight': 0.50,
        'amount_range': (1000000, 50000000),
        'preferred_hours': [0, 1, 2, 3, 4, 23],  # Đêm khuya
        'device_new': True,
        'location_change': True,
        'location_distance': (100, 2000)  # km
    },
    'sim_swap': {  # Chiếm đoạt SIM
        'weight': 0.30,
        'amount_range': (5000000, 100000000),
        'preferred_hours': [9, 10, 14, 15],  # Ngay sau khi swap thành công
        'device_new': True,
        'location_change': True,
        'location_distance': (50, 500)
    },
    'phishing': {  # Lừa đảo website giả
        'weight': 0.20,
        'amount_range': (500000, 20000000),
        'preferred_hours': [19, 20, 21, 22],  # Tối - người dùng check email
        'device_new': True,
        'location_change': True,
        'location_distance': (200, 3000)
    }
}


def round_to_vn_denomination(amount):
    """Làm tròn amount theo mệnh giá VN phổ biến"""
    if amount < 100000:
        # Dưới 100k: làm tròn đến 5k hoặc 10k
        return int(round(amount / 5000) * 5000)
    elif amount < 1000000:
        # 100k - 1M: làm tròn đến 50k hoặc 100k
        return int(round(amount / 50000) * 50000)
    elif amount < 10000000:
        # 1M - 10M: làm tròn đến 500k hoặc 1M
        return int(round(amount / 500000) * 500000)
    else:
        # Trên 10M: làm tròn đến 1M
        return int(round(amount / 1000000) * 1000000)


def generate_realistic_amount(tx_type, is_salary_period=False, is_bill_period=False):
    """Tạo amount thực tế theo loại giao dịch và thời điểm"""

    # Xác định tier dựa trên transaction type
    if tx_type in [0, 6]:  # bill_payment, utilities
        if is_bill_period:
            # Bill điện/nước: 200k-2M
            tier = 'small'
            common_amounts = [200000, 300000, 500000, 700000, 1000000, 1500000]
        else:
            tier = 'micro'
            common_amounts = [50000, 100000, 150000, 200000]
    elif tx_type == 1:  # transfer
        if is_salary_period:
            # Chuyển tiền cuối tháng (tiền thuê nhà, gửi gia đình)
            tier = np.random.choice(['medium', 'large'], p=[0.7, 0.3])
        else:
            tier = np.random.choice(['small', 'medium', 'large'], p=[0.5, 0.4, 0.1])
        common_amounts = TRANSACTION_AMOUNT_CONFIG[tier]['common_amounts']
    elif tx_type in [2, 4]:  # shopping, ecommerce
        tier = np.random.choice(['micro', 'small', 'medium'], p=[0.35, 0.50, 0.15])
        common_amounts = TRANSACTION_AMOUNT_CONFIG[tier]['common_amounts']
    elif tx_type == 3:  # withdrawal
        # ATM thường rút số tròn: 500k, 1M, 2M, 3M, 5M
        return np.random.choice([500000, 1000000, 2000000, 3000000, 5000000],
                               p=[0.25, 0.35, 0.20, 0.10, 0.10])
    elif tx_type == 5:  # food_delivery
        # Grab, ShopeeFood: 30k-200k
        return np.random.choice([35000, 45000, 55000, 65000, 75000, 85000, 100000, 120000, 150000],
                               p=[0.10, 0.15, 0.20, 0.15, 0.15, 0.10, 0.08, 0.05, 0.02])
    elif tx_type == 7:  # investment
        tier = np.random.choice(['medium', 'large'], p=[0.6, 0.4])
        common_amounts = [5000000, 10000000, 20000000, 50000000, 100000000]
    else:
        tier = 'small'
        common_amounts = TRANSACTION_AMOUNT_CONFIG[tier]['common_amounts']

    # 60% chance dùng common amounts, 40% random trong range
    if np.random.random() < 0.6 and common_amounts:
        amount = np.random.choice(common_amounts)
    else:
        config = TRANSACTION_AMOUNT_CONFIG[tier]
        amount = np.random.lognormal(
            np.log((config['min'] + config['max']) / 3),
            0.5
        )
        amount = np.clip(amount, config['min'], config['max'])
        amount = round_to_vn_denomination(amount)

    return int(amount)


def generate_realistic_hour(tx_type, is_weekend=False):
    """Tạo giờ giao dịch thực tế theo loại và ngày"""

    if is_weekend:
        # Cuối tuần: shift giờ muộn hơn, ít bill payment
        if tx_type in [0, 1, 7]:  # bill, transfer, investment
            # Ít giao dịch hơn vào cuối tuần
            hours = [9, 10, 11, 14, 15, 16, 19, 20]
            weights = [0.08, 0.12, 0.12, 0.10, 0.12, 0.12, 0.18, 0.16]
        else:  # shopping, food, ecommerce
            hours = [10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
            weights = [0.08, 0.10, 0.12, 0.08, 0.10, 0.10, 0.10, 0.10, 0.10, 0.08, 0.04]
    else:
        # Ngày thường: peak 9-11h và 14-16h
        if tx_type in [0, 1]:  # bill_payment, transfer
            hours = [8, 9, 10, 11, 14, 15, 16, 17]
            weights = [0.08, 0.15, 0.18, 0.15, 0.12, 0.15, 0.12, 0.05]
        elif tx_type == 5:  # food_delivery
            hours = [11, 12, 13, 18, 19, 20, 21]
            weights = [0.15, 0.25, 0.15, 0.10, 0.15, 0.12, 0.08]
        elif tx_type == 4:  # ecommerce
            hours = [10, 11, 12, 14, 19, 20, 21, 22]
            weights = [0.08, 0.10, 0.08, 0.10, 0.15, 0.20, 0.18, 0.11]
        else:
            hours = [8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
            weights = [0.05, 0.10, 0.12, 0.10, 0.08, 0.08, 0.10, 0.10, 0.07, 0.08, 0.07, 0.05]

    return np.random.choice(hours, p=weights)


def is_salary_period(day_of_month):
    """Kiểm tra có phải kỳ lương không (25-5 hàng tháng)"""
    return day_of_month >= 25 or day_of_month <= 5


def is_bill_period(day_of_month):
    """Kiểm tra có phải kỳ đóng bill không (1-10 hàng tháng)"""
    return 1 <= day_of_month <= 10


# --- Chức năng chính: Tạo DataFrame ---
@st.cache_data
def generate_synthetic_data(num_rows, fraud_ratio, scam_ratio=0.6):
    """
    Tạo dữ liệu synthetic với patterns thực tế Việt Nam

    Args:
        num_rows: Số lượng giao dịch
        fraud_ratio: Tỷ lệ fraud/scam tổng cộng
        scam_ratio: Trong số fraud, tỷ lệ là scam (còn lại là ATO)
    """
    st.info(f"Đang tạo {num_rows:,} dòng dữ liệu với patterns thực tế Việt Nam...")

    # 1. Tạo user base với profiles khác nhau
    num_users = min(5000, num_rows // 4)
    user_profiles = {
        'salary_worker': 0.50,      # Nhân viên văn phòng
        'business_owner': 0.15,     # Chủ doanh nghiệp nhỏ
        'student': 0.15,            # Sinh viên
        'retiree': 0.10,            # Người về hưu
        'freelancer': 0.10          # Freelancer
    }

    user_ids_pool = np.arange(100000, 100000 + num_users)
    user_profile_map = {
        uid: np.random.choice(list(user_profiles.keys()), p=list(user_profiles.values()))
        for uid in user_ids_pool
    }

    # 2. Tạo DataFrame cơ bản
    data = {}
    data['tx_id'] = np.arange(1, num_rows + 1)

    # Gán user_id
    user_ids = np.random.choice(user_ids_pool, num_rows)
    data['user_id'] = user_ids

    # 3. Tạo timestamps với patterns thực tế
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2024-06-30')

    timestamps = []
    for _ in range(num_rows):
        # Random ngày
        days_range = (end_date - start_date).days
        random_day = start_date + pd.Timedelta(days=np.random.randint(0, days_range))

        day_of_month = random_day.day
        day_of_week = random_day.dayofweek
        is_weekend = day_of_week >= 5

        # Tạm thời dùng hour random, sẽ adjust sau
        hour = np.random.randint(6, 23)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)

        ts = random_day.replace(hour=hour, minute=minute, second=second)
        timestamps.append(ts)

    data['timestamp'] = timestamps

    df = pd.DataFrame(data).sort_values(by='timestamp').reset_index(drop=True)

    # 4. Tạo transaction_type dựa trên ngày và user profile
    tx_types = []
    for idx, row in df.iterrows():
        ts = row['timestamp']
        user_id = row['user_id']
        profile = user_profile_map[user_id]

        day_of_month = ts.day
        is_weekend = ts.dayofweek >= 5

        # Weights dựa trên profile và thời điểm
        if profile == 'salary_worker':
            if is_bill_period(day_of_month):
                weights = [0.25, 0.20, 0.15, 0.10, 0.15, 0.10, 0.03, 0.02]
            elif is_salary_period(day_of_month):
                weights = [0.10, 0.30, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05]
            else:
                weights = [0.05, 0.15, 0.20, 0.10, 0.25, 0.15, 0.05, 0.05]
        elif profile == 'business_owner':
            weights = [0.10, 0.35, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10]
        elif profile == 'student':
            weights = [0.05, 0.10, 0.15, 0.05, 0.35, 0.25, 0.05, 0.00]
        elif profile == 'retiree':
            weights = [0.25, 0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.00]
        else:  # freelancer
            weights = [0.10, 0.25, 0.15, 0.10, 0.20, 0.10, 0.05, 0.05]

        tx_type = np.random.choice(list(TRANSACTION_TYPES.keys()), p=weights)
        tx_types.append(tx_type)

    df['transaction_type'] = tx_types

    # 5. Tạo amounts thực tế
    amounts = []
    for idx, row in df.iterrows():
        ts = row['timestamp']
        tx_type = row['transaction_type']

        day_of_month = ts.day
        salary_period = is_salary_period(day_of_month)
        bill_period = is_bill_period(day_of_month)

        amount = generate_realistic_amount(tx_type, salary_period, bill_period)
        amounts.append(amount)

    df['amount'] = amounts

    # 6. Adjust hours based on transaction type
    adjusted_timestamps = []
    for idx, row in df.iterrows():
        ts = row['timestamp']
        tx_type = row['transaction_type']
        is_weekend = ts.dayofweek >= 5

        new_hour = generate_realistic_hour(tx_type, is_weekend)
        new_ts = ts.replace(hour=new_hour)
        adjusted_timestamps.append(new_ts)

    df['timestamp'] = adjusted_timestamps
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 7. Các derived columns từ timestamp
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_salary_period'] = df['day_of_month'].apply(is_salary_period).astype(int)
    df['is_bill_period'] = df['day_of_month'].apply(is_bill_period).astype(int)
    df['is_night_hours'] = df['hour_of_day'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

    # 8. Amount derived features
    df['amount_log'] = np.log1p(df['amount'])
    # Bỏ amount_norm để tránh multicollinearity với amount_log
    df['amount_percentile_system'] = df['amount'].rank(pct=True)

    # 9. Account và device features
    df['account_age_days'] = np.random.randint(30, 365 * 5, num_rows)
    df['channel'] = np.random.choice([0, 1, 2], num_rows, p=[0.70, 0.20, 0.10])  # 0: Mobile, 1: Web, 2: ATM
    df['location_diff_km'] = np.clip(np.random.lognormal(0.5, 1, num_rows) - 0.5, 0, 100)  # Bình thường < 50km

    # 10. Behavioral features
    df['time_gap_prev_min'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0) / 60
    df['time_gap_prev_min'] = df['time_gap_prev_min'].apply(lambda x: x if x > 1 else np.random.lognormal(3, 1))

    # Velocity calculations
    df_temp = df.set_index('timestamp')
    velocity_1h_series = df_temp.groupby('user_id')['tx_id'].rolling('1h', closed='left').count()
    velocity_24h_series = df_temp.groupby('user_id')['tx_id'].rolling('24h', closed='left').count()

    df_velocity = velocity_1h_series.rename('velocity_1h').to_frame()
    df_velocity['velocity_24h'] = velocity_24h_series
    df = df.merge(df_velocity.reset_index(), on=['user_id', 'timestamp'], how='left')

    df['velocity_1h'] = df['velocity_1h'].fillna(0)
    df['velocity_24h'] = df['velocity_24h'].fillna(0)
    # Bỏ freq_norm để tránh multicollinearity với velocity_24h

    # 11. Recipient và device patterns
    df['is_new_recipient'] = np.random.choice([0, 1], num_rows, p=[0.88, 0.12])
    df['recipient_count_30d'] = np.clip(np.random.lognormal(1.2, 0.5, num_rows).astype(int), 1, 20)
    df['is_new_device'] = np.random.choice([0, 1], num_rows, p=[0.96, 0.04])
    df['device_count_30d'] = np.clip(np.random.lognormal(0.5, 0.3, num_rows).astype(int) + 1, 1, 4)

    # 12. Scam detection features
    # Amount vs user average
    df_temp2 = df.set_index('timestamp')
    avg_amount_series = df_temp2.groupby('user_id')['amount'].rolling('30d', closed='left').mean()
    avg_amount_series = avg_amount_series.shift(1).fillna(df['amount'].mean() * 0.5)
    df_avg = avg_amount_series.rename('avg_amount_30d').to_frame().reset_index()
    df = df.merge(df_avg, on=['user_id', 'timestamp'], how='left')

    df['amount_vs_avg_user'] = df['amount'] / df['avg_amount_30d'].clip(lower=10000)
    df.drop(columns=['avg_amount_30d'], inplace=True)

    # Is first large transaction
    df['max_amount_seen'] = df.groupby('user_id')['amount'].transform(lambda x: x.cummax().shift(1).fillna(0))
    df['is_first_large_tx'] = ((df['amount'] > 2 * df['max_amount_seen']) & (df['max_amount_seen'] > 0)).astype(int)
    df.drop(columns=['max_amount_seen'], inplace=True)

    # Suspicious recipient
    df['recipient_is_suspicious'] = np.random.choice([0, 1], num_rows, p=[0.995, 0.005])

    # 13. Tạo labels: fraud/scam
    num_fraud_total = int(num_rows * fraud_ratio)
    num_scam = int(num_fraud_total * scam_ratio)
    num_ato = num_fraud_total - num_scam

    labels = np.zeros(num_rows, dtype=int)
    fraud_type = ['normal'] * num_rows  # Track loại fraud

    # Chọn indices cho fraud
    all_fraud_indices = np.random.choice(num_rows, num_fraud_total, replace=False)
    scam_indices = all_fraud_indices[:num_scam]
    ato_indices = all_fraud_indices[num_scam:]

    labels[all_fraud_indices] = 1
    df['is_fraud'] = labels

    # 14. Điều chỉnh data cho SCAM patterns (Lừa đảo - nạn nhân tự chuyển)
    scam_types = list(SCAM_PATTERNS.keys())
    scam_weights = [SCAM_PATTERNS[s]['weight'] for s in scam_types]

    for idx in scam_indices:
        scam_type = np.random.choice(scam_types, p=scam_weights)
        pattern = SCAM_PATTERNS[scam_type]
        fraud_type[idx] = f'scam_{scam_type}'

        # Amount theo pattern
        amount_min, amount_max = pattern['amount_range']
        if scam_type == 'impersonation':
            # "Công an gọi": số tiền lớn, thường là số tròn
            amount = np.random.choice([10000000, 20000000, 30000000, 50000000, 70000000, 100000000])
        elif scam_type == 'job_scam':
            # "Việc làm online": ban đầu nhỏ, sau lớn dần
            if np.random.random() < 0.7:
                amount = np.random.choice([100000, 200000, 300000, 500000])  # "Phí đăng ký"
            else:
                amount = np.random.choice([2000000, 3000000, 5000000])  # "Nâng cấp"
        elif scam_type == 'lottery_prize':
            # "Trúng thưởng": phí nhỏ nhiều lần
            amount = np.random.choice([500000, 700000, 1000000, 1500000, 2000000])
        elif scam_type == 'romance_scam':
            # Lừa tình: số tiền lớn, cảm xúc
            amount = np.random.choice([5000000, 10000000, 15000000, 20000000, 30000000, 50000000])
        else:  # investment_scam
            # Đầu tư lừa đảo: tăng dần
            amount = np.random.choice([2000000, 5000000, 10000000, 20000000, 50000000, 100000000])

        df.loc[idx, 'amount'] = amount

        # Hour theo pattern
        df.loc[idx, 'hour_of_day'] = np.random.choice(pattern['preferred_hours'])

        # Scam: device và location thường bình thường (nạn nhân tự chuyển)
        df.loc[idx, 'is_new_device'] = 0
        df.loc[idx, 'location_diff_km'] = np.random.uniform(0, 10)

        # Nhưng recipient mới và suspicious
        df.loc[idx, 'is_new_recipient'] = 1
        df.loc[idx, 'recipient_is_suspicious'] = 1

        # Amount vs avg cao bất thường
        df.loc[idx, 'amount_vs_avg_user'] = np.random.uniform(3, 15)
        df.loc[idx, 'is_first_large_tx'] = 1

        # Transaction type: chủ yếu là transfer
        df.loc[idx, 'transaction_type'] = 1

    # 15. Điều chỉnh data cho ATO patterns (Hacker chiếm tài khoản)
    ato_types = list(ATO_PATTERNS.keys())
    ato_weights = [ATO_PATTERNS[a]['weight'] for a in ato_types]

    for idx in ato_indices:
        ato_type = np.random.choice(ato_types, p=ato_weights)
        pattern = ATO_PATTERNS[ato_type]
        fraud_type[idx] = f'ato_{ato_type}'

        # Amount
        amount_min, amount_max = pattern['amount_range']
        amount = np.random.lognormal(np.log((amount_min + amount_max) / 3), 0.8)
        amount = np.clip(amount, amount_min, amount_max)
        df.loc[idx, 'amount'] = round_to_vn_denomination(amount)

        # Hour - thường đêm khuya
        df.loc[idx, 'hour_of_day'] = np.random.choice(pattern['preferred_hours'])
        df.loc[idx, 'is_night_hours'] = 1 if df.loc[idx, 'hour_of_day'] in [22, 23, 0, 1, 2, 3, 4, 5] else 0

        # ATO: device mới, location khác
        df.loc[idx, 'is_new_device'] = 1

        loc_min, loc_max = pattern['location_distance']
        df.loc[idx, 'location_diff_km'] = np.random.uniform(loc_min, loc_max)

        # Recipient mới
        df.loc[idx, 'is_new_recipient'] = 1

        # Time gap ngắn (hacker muốn nhanh)
        df.loc[idx, 'time_gap_prev_min'] = np.random.uniform(0.5, 5)

        # Velocity cao
        df.loc[idx, 'velocity_1h'] = np.random.uniform(3, 10)
        df.loc[idx, 'velocity_24h'] = np.random.uniform(5, 15)

        # Transaction type: chủ yếu transfer hoặc withdrawal
        df.loc[idx, 'transaction_type'] = np.random.choice([1, 3], p=[0.7, 0.3])

    df['fraud_type'] = fraud_type

    # 16. Recalculate derived columns sau khi modify
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_percentile_system'] = df['amount'].rank(pct=True)

    # 17. Thêm features tối ưu cho từng model

    # === Features cho ISOLATION FOREST (unsupervised anomaly detection) ===
    # Isolation Forest hiệu quả với features có phân bố continuous, ít categorical

    # Amount deviation từ median user (robust hơn mean)
    user_median_amount = df.groupby('user_id')['amount'].transform('median')
    df['amount_deviation_ratio'] = df['amount'] / user_median_amount.clip(lower=10000)

    # Time since last transaction (hours) - outlier indicator
    df['hours_since_prev_tx'] = df['time_gap_prev_min'] / 60

    # Velocity ratio (1h vs 24h) - burst detection
    df['velocity_ratio'] = df['velocity_1h'] / (df['velocity_24h'].clip(lower=1))

    # Location anomaly score (log scale cho outlier detection)
    df['location_anomaly'] = np.log1p(df['location_diff_km'])

    # Hour deviation from user's typical pattern
    user_typical_hour = df.groupby('user_id')['hour_of_day'].transform('median')
    df['hour_deviation'] = np.abs(df['hour_of_day'] - user_typical_hour)

    # Account age risk (younger = riskier)
    df['account_age_risk'] = 1 / np.log1p(df['account_age_days'])

    # === Features cho LIGHTGBM (supervised classification) ===
    # LightGBM xử lý tốt categorical và interactions

    # Transaction type risk encoding (dựa trên fraud rate thực tế VN)
    tx_risk_map = {0: 0.02, 1: 0.15, 2: 0.03, 3: 0.08, 4: 0.05, 5: 0.01, 6: 0.02, 7: 0.12}
    df['tx_type_risk'] = df['transaction_type'].map(tx_risk_map)

    # Channel risk encoding
    channel_risk_map = {0: 0.04, 1: 0.08, 2: 0.06}  # Mobile, Web, ATM
    df['channel_risk'] = df['channel'].map(channel_risk_map)

    # Combined behavioral risk score
    df['behavioral_risk_score'] = (
        df['is_new_recipient'] * 0.25 +
        df['is_new_device'] * 0.30 +
        df['is_night_hours'] * 0.15 +
        df['recipient_is_suspicious'] * 0.30
    )

    # Amount tier encoding (ordinal)
    df['amount_tier'] = pd.cut(df['amount'],
                               bins=[0, 200000, 2000000, 20000000, float('inf')],
                               labels=[0, 1, 2, 3]).astype(int)

    # Time context risk (weekend + night combo)
    df['time_context_risk'] = df['is_weekend'] * 0.3 + df['is_night_hours'] * 0.7

    # User activity level (transactions per day estimate)
    df['user_activity_level'] = df.groupby('user_id')['tx_id'].transform('count') / 180  # 6 months

    # Recipient diversity risk
    df['recipient_diversity'] = df['recipient_count_30d'] / 30  # Normalized

    # 18. Global anomaly score (cập nhật với features mới)
    base_score = (
        df['amount_percentile_system'] * 2 +
        df['is_night_hours'] * 1.5 +
        df['is_new_recipient'] * 1.5 +
        df['is_new_device'] * 2 +
        (df['location_diff_km'].clip(upper=500) / 500) * 2 +
        (df['amount_vs_avg_user'].clip(upper=10) / 10) * 2 +
        df['recipient_is_suspicious'] * 3 +
        (1 - df['time_gap_prev_min'].clip(upper=60) / 60) * 1 +
        df['velocity_1h'].clip(upper=5) / 5 * 1
    )

    df['global_anomaly_score_prev'] = (base_score + np.random.normal(0, 0.3, num_rows)) / base_score.max()
    df['global_anomaly_score_prev'] = df['global_anomaly_score_prev'].clip(0.01, 0.99)

    # Boost score cho fraud cases
    df.loc[df['is_fraud'] == 1, 'global_anomaly_score_prev'] = np.clip(
        df.loc[df['is_fraud'] == 1, 'global_anomaly_score_prev'] + np.random.uniform(0.15, 0.35, num_fraud_total),
        0.65, 0.99
    )

    # 19. Chọn final columns
    final_columns = [
        # Identification
        'tx_id', 'user_id',
        # Transaction info
        'transaction_type', 'amount', 'channel',
        # Amount features
        'amount_log', 'amount_percentile_system', 'amount_vs_avg_user', 'amount_tier',
        # Time features
        'hour_of_day', 'day_of_week', 'day_of_month',
        'is_weekend', 'is_salary_period', 'is_bill_period', 'is_night_hours',
        # Behavioral features
        'time_gap_prev_min', 'velocity_1h', 'velocity_24h',
        'is_new_recipient', 'recipient_count_30d', 'is_new_device', 'device_count_30d',
        # Location features
        'location_diff_km', 'account_age_days',
        # Risk indicators
        'is_first_large_tx', 'recipient_is_suspicious',
        # Isolation Forest optimized features
        'amount_deviation_ratio', 'hours_since_prev_tx', 'velocity_ratio',
        'location_anomaly', 'hour_deviation', 'account_age_risk',
        # LightGBM optimized features
        'tx_type_risk', 'channel_risk', 'behavioral_risk_score',
        'time_context_risk', 'user_activity_level', 'recipient_diversity',
        # Scores and labels
        'global_anomaly_score_prev', 'fraud_type', 'is_fraud'
    ]

    df_output = df[final_columns].copy()

    # Statistics
    num_scam_actual = len([f for f in fraud_type if f.startswith('scam_')])
    num_ato_actual = len([f for f in fraud_type if f.startswith('ato_')])

    st.success(f"""
    Tạo dữ liệu thành công!
    - Tổng giao dịch: {num_rows:,}
    - Fraud/Scam: {df_output['is_fraud'].sum():,} ({df_output['is_fraud'].mean()*100:.2f}%)
    - Trong đó SCAM (lừa đảo): {num_scam_actual:,}
    - Trong đó ATO (hacker): {num_ato_actual:,}
    """)

    return df_output.sort_values(by='tx_id').reset_index(drop=True)


# --- Giao diện Streamlit ---
st.title("🛡️ Công cụ Tạo Dữ liệu Giao Dịch - Vietnam Fraud & Scam Detection")

st.markdown("""
### Đặc điểm dữ liệu
- **Amount thực tế VN**: Phân bố theo loại giao dịch (cà phê 25-50k, bill 200k-2M, chuyển tiền 2M-20M...)
- **Time patterns VN**: Peak 9-11h & 14-16h, tăng GD ngày 25-5 (lương), bill ngày 1-10
- **Scam patterns VN**: Công an gọi, việc làm online, trúng thưởng, romance scam, investment scam
- **ATO patterns**: Credential theft, SIM swap, Phishing
""")

st.sidebar.header("⚙️ Cấu hình Dữ liệu")

num_rows_target = st.sidebar.number_input(
    "Số lượng giao dịch",
    min_value=1000,
    max_value=500000,
    value=50000,
    step=5000,
    help="Khuyến nghị 50,000+ dòng để train model hiệu quả"
)

fraud_ratio_target = st.sidebar.slider(
    "Tỷ lệ Fraud/Scam tổng (%)",
    min_value=0.5,
    max_value=15.0,
    value=5.0,
    step=0.5,
    format="%.1f%%",
    help="Tỷ lệ thực tế VN khoảng 2-5%"
)

scam_ratio_target = st.sidebar.slider(
    "Trong Fraud, tỷ lệ SCAM vs ATO (%)",
    min_value=20.0,
    max_value=80.0,
    value=60.0,
    step=5.0,
    format="%.0f%% SCAM",
    help="VN: SCAM (lừa đảo) phổ biến hơn ATO (hack)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Giải thích:**
- **SCAM**: Nạn nhân bị lừa tự chuyển tiền
- **ATO**: Hacker chiếm tài khoản để chuyển
""")

if st.sidebar.button("🚀 Tạo Dữ liệu", use_container_width=True):
    with st.spinner("Đang tạo dữ liệu..."):
        df_generated = generate_synthetic_data(
            num_rows_target,
            fraud_ratio_target / 100,
            scam_ratio_target / 100
        )
        st.session_state['generated_data'] = df_generated

st.markdown("---")

if 'generated_data' in st.session_state:
    df_display = st.session_state['generated_data']

    st.subheader("📊 Dữ liệu đã tạo")
    st.dataframe(df_display, use_container_width=True)

    st.subheader("📈 Thống kê tổng quan")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng giao dịch", f"{len(df_display):,}")
    with col2:
        st.metric("Số Fraud/Scam", f"{df_display['is_fraud'].sum():,}")
    with col3:
        st.metric("Tỷ lệ Fraud", f"{df_display['is_fraud'].mean()*100:.2f}%")
    with col4:
        st.metric("Số người dùng", f"{df_display['user_id'].nunique():,}")

    # Thống kê chi tiết
    st.subheader("📊 Phân tích chi tiết")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Phân bố loại giao dịch:**")
        tx_type_counts = df_display['transaction_type'].value_counts().sort_index()
        tx_type_names = {i: name for i, name in TRANSACTION_TYPES.items()}
        tx_summary = pd.DataFrame({
            'Loại': [tx_type_names.get(i, f'Type {i}') for i in tx_type_counts.index],
            'Số lượng': tx_type_counts.values,
            'Tỷ lệ': [f"{v/len(df_display)*100:.1f}%" for v in tx_type_counts.values]
        })
        st.dataframe(tx_summary, use_container_width=True)

    with col2:
        st.markdown("**Phân bố loại Fraud:**")
        fraud_types = df_display[df_display['is_fraud'] == 1]['fraud_type'].value_counts()
        fraud_summary = pd.DataFrame({
            'Loại': fraud_types.index,
            'Số lượng': fraud_types.values,
            'Tỷ lệ': [f"{v/len(fraud_types)*100:.1f}%" for v in fraud_types.values]
        })
        st.dataframe(fraud_summary, use_container_width=True)

    # Amount statistics
    st.markdown("**Thống kê Amount (VND):**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min", f"{df_display['amount'].min():,.0f}")
    with col2:
        st.metric("Median", f"{df_display['amount'].median():,.0f}")
    with col3:
        st.metric("Mean", f"{df_display['amount'].mean():,.0f}")
    with col4:
        st.metric("Max", f"{df_display['amount'].max():,.0f}")

    st.subheader("📥 Tải xuống dữ liệu")

    # Định nghĩa features cho từng model
    # ISOLATION FOREST: Unsupervised - KHÔNG dùng label để train
    ISOLATION_FOREST_FEATURES = [
        'amount_log', 'amount_deviation_ratio', 'amount_vs_avg_user',
        'hours_since_prev_tx', 'velocity_1h', 'velocity_24h', 'velocity_ratio',
        'location_diff_km', 'location_anomaly',
        'hour_deviation', 'is_night_hours',
        'is_new_recipient', 'is_new_device',
        'account_age_risk'
        # KHÔNG có is_fraud - Isolation Forest là unsupervised
    ]

    # LIGHTGBM: Supervised - CẦN label để train
    LIGHTGBM_FEATURES = [
        'transaction_type', 'amount_log', 'amount_tier', 'amount_vs_avg_user',
        'channel', 'channel_risk', 'tx_type_risk',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_night_hours',
        'is_salary_period', 'is_bill_period',
        'time_gap_prev_min', 'velocity_1h', 'velocity_24h',
        'is_new_recipient', 'recipient_count_30d', 'is_new_device', 'device_count_30d',
        'location_diff_km', 'account_age_days',
        'is_first_large_tx', 'recipient_is_suspicious',
        'behavioral_risk_score', 'time_context_risk',
        'user_activity_level', 'recipient_diversity',
        'is_fraud'  # Label - LightGBM là supervised, CẦN label
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Full Data (Phân tích)**")
        # Full data
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Full Data (CSV)",
            data=csv_data,
            file_name=f"vietnam_fraud_full_{len(df_display)}_rows.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"{len(df_display.columns)} features - Đầy đủ để phân tích")

    with col2:
        st.markdown("**Training Data (Không fraud_type)**")
        train_cols = [c for c in df_display.columns if c != 'fraud_type']
        csv_train = df_display[train_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Training Data (CSV)",
            data=csv_train,
            file_name=f"vietnam_fraud_train_{len(df_display)}_rows.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"{len(train_cols)} features - Không có fraud_type column")

    st.markdown("---")
    st.markdown("**Training Data cho từng Model:**")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Isolation Forest** (Unsupervised)")
        st.warning("⚠️ **KHÔNG dùng label để train!** File chỉ chứa features.")

        # File features (để train)
        df_isolation = df_display[ISOLATION_FOREST_FEATURES].copy()
        csv_isolation = df_isolation.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ IF Features (để Train)",
            data=csv_isolation,
            file_name=f"vietnam_IF_features_{len(df_display)}_rows.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"{len(ISOLATION_FOREST_FEATURES)} features - Dùng để train model")

        # File labels riêng (để evaluate)
        df_labels = df_display[['is_fraud', 'fraud_type']].copy()
        csv_labels = df_labels.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Labels (để Evaluate)",
            data=csv_labels,
            file_name=f"vietnam_IF_labels_{len(df_display)}_rows.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Labels riêng - Dùng để đánh giá model sau khi train")

        with st.expander("Xem danh sách features"):
            for f in ISOLATION_FOREST_FEATURES:
                st.markdown(f"- `{f}`")

    with col4:
        st.markdown("**LightGBM** (Supervised)")
        st.info("ℹ️ **CẦN label để train!** File bao gồm is_fraud.")

        df_lgbm = df_display[LIGHTGBM_FEATURES].copy()
        csv_lgbm = df_lgbm.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ LightGBM Data (Features + Label)",
            data=csv_lgbm,
            file_name=f"vietnam_LGBM_train_{len(df_display)}_rows.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"{len(LIGHTGBM_FEATURES)} columns (29 features + 1 label)")

        with st.expander("Xem danh sách features"):
            for f in LIGHTGBM_FEATURES[:-1]:  # Trừ is_fraud
                st.markdown(f"- `{f}`")
            st.markdown("- `is_fraud` *(label)*")

else:
    st.info("👈 Cấu hình tham số ở sidebar và nhấn **Tạo Dữ liệu** để bắt đầu.")

    tab1, tab2, tab3 = st.tabs(["Tổng quan", "Isolation Forest", "LightGBM"])

    with tab1:
        st.markdown("""
        ### Các features chính:

        | Feature | Mô tả |
        |---------|-------|
        | `transaction_type` | 0: Bill, 1: Transfer, 2: Shopping, 3: ATM, 4: Ecommerce, 5: Food, 6: Utilities, 7: Investment |
        | `amount_log` | Log-transformed amount (tốt hơn cho ML) |
        | `is_salary_period` | 1 nếu ngày 25-5 (kỳ lương) |
        | `is_bill_period` | 1 nếu ngày 1-10 (kỳ đóng bill) |
        | `is_night_hours` | 1 nếu 22h-5h (giờ bất thường) |
        | `amount_vs_avg_user` | Tỷ lệ so với trung bình 30 ngày của user |
        | `fraud_type` | Loại fraud: scam_impersonation, scam_job_scam, ato_credential_theft... |
        """)

    with tab2:
        st.markdown("""
        ### Features cho Isolation Forest (14 features)

        **Isolation Forest** là **unsupervised** anomaly detection:
        - ⚠️ **KHÔNG dùng label (`is_fraud`) để train**
        - Label chỉ dùng để **evaluate** model sau khi train
        - Hoạt động bằng cách "cô lập" các điểm bất thường qua random splits

        #### Danh sách features:

        | Feature | Loại | Mô tả |
        |---------|------|-------|
        | `amount_log` | Continuous | Log của số tiền (giảm skewness) |
        | `amount_deviation_ratio` | Continuous | Tỷ lệ so với median của user |
        | `amount_vs_avg_user` | Continuous | Tỷ lệ so với mean 30 ngày của user |
        | `hours_since_prev_tx` | Continuous | Số giờ kể từ GD trước (hours) |
        | `velocity_1h` | Continuous | Số GD trong 1 giờ qua |
        | `velocity_24h` | Continuous | Số GD trong 24 giờ qua |
        | `velocity_ratio` | Continuous | Tỷ lệ velocity 1h/24h (burst detection) |
        | `location_diff_km` | Continuous | Khoảng cách từ vị trí thường (km) |
        | `location_anomaly` | Continuous | Log của location_diff_km |
        | `hour_deviation` | Continuous | Độ lệch so với giờ thường của user |
        | `is_night_hours` | Binary | 1 nếu 22h-5h |
        | `is_new_recipient` | Binary | 1 nếu người nhận mới |
        | `is_new_device` | Binary | 1 nếu thiết bị mới |
        | `account_age_risk` | Continuous | 1/log(account_age) - tài khoản mới rủi ro hơn |

        #### Cách sử dụng đúng:
        ```python
        # 1. Load features (KHÔNG có label)
        X = pd.read_csv('vietnam_IF_features_50000_rows.csv')

        # 2. Train Isolation Forest (unsupervised)
        model = IsolationForest(contamination=0.05)
        model.fit(X)  # Không có y!

        # 3. Predict
        predictions = model.predict(X)  # -1 = anomaly

        # 4. Evaluate với labels riêng
        labels = pd.read_csv('vietnam_IF_labels_50000_rows.csv')
        ```
        """)

    with tab3:
        st.markdown("""
        ### Features cho LightGBM (29 features)

        **LightGBM** là supervised gradient boosting - cần label để train.
        Xử lý tốt categorical features và feature interactions.

        | Feature | Loại | Mô tả |
        |---------|------|-------|
        | `transaction_type` | Categorical | Loại giao dịch (0-7) |
        | `amount_log` | Continuous | Log của số tiền |
        | `amount_tier` | Ordinal | Mức amount (0: <200k, 1: 200k-2M, 2: 2M-20M, 3: >20M) |
        | `amount_vs_avg_user` | Continuous | Tỷ lệ so với trung bình user |
        | `channel` | Categorical | 0: Mobile, 1: Web, 2: ATM |
        | `channel_risk` | Continuous | Risk score theo channel |
        | `tx_type_risk` | Continuous | Risk score theo loại GD |
        | `hour_of_day` | Continuous | Giờ trong ngày (0-23) |
        | `day_of_week` | Categorical | Ngày trong tuần (0-6) |
        | `is_weekend` | Binary | 1 nếu thứ 7/CN |
        | `is_night_hours` | Binary | 1 nếu 22h-5h |
        | `is_salary_period` | Binary | 1 nếu ngày 25-5 |
        | `is_bill_period` | Binary | 1 nếu ngày 1-10 |
        | `time_gap_prev_min` | Continuous | Phút kể từ GD trước |
        | `velocity_1h` | Continuous | Số GD trong 1 giờ |
        | `velocity_24h` | Continuous | Số GD trong 24 giờ |
        | `is_new_recipient` | Binary | 1 nếu người nhận mới |
        | `recipient_count_30d` | Continuous | Số người nhận trong 30 ngày |
        | `is_new_device` | Binary | 1 nếu thiết bị mới |
        | `device_count_30d` | Continuous | Số thiết bị trong 30 ngày |
        | `location_diff_km` | Continuous | Khoảng cách từ vị trí thường |
        | `account_age_days` | Continuous | Tuổi tài khoản (ngày) |
        | `is_first_large_tx` | Binary | 1 nếu là GD lớn đầu tiên |
        | `recipient_is_suspicious` | Binary | 1 nếu người nhận đáng ngờ |
        | `behavioral_risk_score` | Continuous | Combined risk từ behavior |
        | `time_context_risk` | Continuous | Risk từ weekend + night |
        | `user_activity_level` | Continuous | Mức hoạt động của user |
        | `recipient_diversity` | Continuous | Đa dạng người nhận |
        """)
