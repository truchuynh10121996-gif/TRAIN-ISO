# streamlit_generate_datasets.py
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Synthetic Fraud Dataset Generator", layout="wide")
st.title("Synthetic Fraud Dataset Generator — Isolation & LightGBM")
st.markdown("""
Ứng dụng tạo file CSV mẫu để huấn luyện **IsolationForest (unsupervised)** và **LightGBM (supervised)** cho bài toán phát hiện gian lận.
- **21 cột** (behavioral + time + device + network features).  
- **Amount**: làm tròn **nghìn VND**.  
- Mục tiêu: dữ liệu mô phỏng hợp thực tế ~85–95% về phân phối hành vi (khả năng dùng làm POC/POV).
""")

# --------------------------
# Controls
# --------------------------
col1, col2 = st.columns(2)
with col1:
    n_rows = st.number_input("Số dòng (rows)", min_value=1000, max_value=200000, value=20000, step=1000)
    fraud_rate = st.slider("Tỉ lệ fraud (cho dataset Isolation)", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
    seed = st.number_input("Random seed", value=42)
    round_thousand = st.checkbox("Làm tròn tiền theo nghìn (bắt buộc)", value=True)
with col2:
    n_users = st.number_input("Số lượng user giả lập", min_value=100, max_value=50000, value=3000, step=100)
    merchant_ratio = st.slider("Tỉ lệ merchant trong user pool", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
    realisticness = st.slider("Độ sát thực tế (0.7=70% ... 0.99=99%)", min_value=0.7, max_value=0.99, value=0.9, step=0.01)

st.markdown("**Tùy chọn nâng cao** (mặc định tốt cho POC).")
with st.expander("Tùy chọn advanced"):
    avg_tx_per_user = st.number_input("Trung bình giao dịch / user trong 60 ngày (approx)", min_value=1, max_value=200, value=7)
    include_labels_for_lgb = st.checkbox("Kèm nhãn fraud cho LightGBM (supervised)", value=True)
    min_amount = st.number_input("Số tiền giao dịch tối thiểu (VND)", value=1000, step=1000)
    max_amount = st.number_input("Số tiền giao dịch tối đa cho normal users (VND)", value=2000000, step=1000)

# --------------------------
# Helper functions
# --------------------------
def round_to_thousand(x: float) -> int:
    return int(round(x/1000.0) * 1000)

def make_user_profiles(n_users: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    user_ids = [f"U{str(i).zfill(6)}" for i in range(1, n_users+1)]
    profiles = {}
    for uid in user_ids:
        # Mix of low/medium/high spenders -> gives realistic distribution
        t = np.random.choice(['low','medium','high'], p=[0.65, 0.28, 0.07])
        if t == 'low':
            avg = int(np.random.choice([20000,50000,80000]))
        elif t == 'medium':
            avg = int(np.random.choice([150000,300000,500000]))
        else:
            avg = int(np.random.choice([1000000,2000000]))
        std = max(1000, int(avg * np.random.uniform(0.18, 0.6)))
        account_age_days = int(np.random.exponential(scale=500)) + 30
        profiles[uid] = {'avg': avg, 'std': std, 'account_age_days': account_age_days}
    return profiles

def sample_timestamp(last_60_days=True):
    base = datetime.now() - timedelta(days=60) if last_60_days else datetime.now() - timedelta(days=365)
    ts = base + timedelta(seconds=random.randint(0, 60*24*3600))
    return ts

# --------------------------
# Main generation
# --------------------------
if st.button("Generate datasets"):
    st.info("Đang tạo dữ liệu... (Xin chờ vài giây)")
    random.seed(int(seed)); np.random.seed(int(seed))

    # Build pools
    user_ids = [f"U{str(i).zfill(6)}" for i in range(1, int(n_users)+1)]
    n_merchants = max(1, int(len(user_ids) * merchant_ratio))
    merchant_ids = [f"M{str(i).zfill(5)}" for i in range(1, n_merchants+1)]
    external_ids = [f"R{str(i).zfill(6)}" for i in range(1, max(500, int(len(user_ids)*0.1)))]
    fraud_ids_pool = [f"F{str(i).zfill(6)}" for i in range(1, 2000)]

    profiles = make_user_profiles(len(user_ids), int(seed))

    total_rows = int(n_rows)
    n_fraud = int(total_rows * fraud_rate)
    n_normal = total_rows - n_fraud

    rows = []
    tx_counter = 1

    def create_row(sender, receiver, amount, ts, is_new_recipient, is_new_device, device_count_30d,
                   location_diff_km, channel, is_fraud_flag):
        nonlocal tx_counter
        tx_id = f"TX{str(tx_counter).zfill(10)}"
        tx_counter += 1
        hour = ts.hour
        dow = ts.weekday()
        is_weekend = 1 if dow >= 5 else 0
        return {
            "tx_id": tx_id,
            "user_id": sender,
            "amount": int(amount),
            "amount_log": float(np.log(amount + 1)),
            "amount_norm": None, # fill later
            "hour_of_day": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "time_gap_prev_min": None, # fill later
            "velocity_1h": None,
            "velocity_24h": None,
            "freq_norm": None,
            "is_new_recipient": int(is_new_recipient),
            "recipient_count_30d": None,
            "is_new_device": int(is_new_device),
            "device_count_30d": device_count_30d,
            "location_diff_km": float(location_diff_km),
            "channel": channel,
            "account_age_days": int(profiles[sender]['account_age_days']),
            "amount_percentile_system": None,
            "global_anomaly_score_prev": float(np.random.uniform(0.6,1.0)) if is_fraud_flag else float(np.random.uniform(0.0,0.3)),
            "is_fraud": int(is_fraud_flag)
        }

    # Generate normal transactions
    for _ in range(n_normal):
        sender = random.choice(user_ids)
        receiver = random.choice(user_ids + merchant_ids + external_ids)
        prof = profiles[sender]
        # amount around user's avg with some noise; clip to sensible range
        amount = int(np.random.normal(loc=prof['avg'], scale=prof['std']))
        amount = max(int(min_amount), amount)
        if amount > max_amount:
            # realistic: most users rarely exceed max_amount; we cap
            amount = int(np.random.uniform(prof['avg']*0.6, max_amount))
        if round_thousand:
            amount = round_to_thousand(amount)
        ts = sample_timestamp()
        # bias to daytime (realistic behaviour)
        if random.random() < realisticness:
            ts = ts.replace(hour=random.randint(7,21))
        is_new_recipient = random.random() < 0.03
        is_new_device = random.random() < 0.02
        device_count_30d = max(1, int(np.random.poisson(1) + 1))
        location_diff_km = round(abs(np.random.normal(5, 8)), 1)  # many in-city small distance
        channel = np.random.choice(["mobile", "web", "atm"], p=[0.75, 0.20, 0.05])
        rows.append(create_row(sender, receiver, amount, ts, is_new_recipient, is_new_device, device_count_30d, location_diff_km, channel, False))

    # Generate fraud transactions (suspicious patterns)
    for _ in range(n_fraud):
        sender = random.choice(user_ids)
        receiver = random.choice(merchant_ids + external_ids + fraud_ids_pool)
        prof = profiles[sender]
        # fraud amounts skewed higher or unusual
        if random.random() < 0.6:
            amount = int(np.random.choice([3000000, 4990000, 5000000, 9000000, 10000000]))
        else:
            amount = max(int(min_amount), int(np.random.normal(loc=max(prof['avg'] * 6, 2000000), scale=prof['std']*2)))
        if round_thousand:
            amount = round_to_thousand(amount)
        ts = sample_timestamp()
        # fraud often off-hours / clustered
        if random.random() < 0.6:
            ts = ts.replace(hour=random.randint(0,6))
        is_new_recipient = True if random.random() < 0.8 else False
        is_new_device = True if random.random() < 0.6 else False
        device_count_30d = max(1, int(np.random.poisson(3) + 1))
        location_diff_km = round(abs(np.random.normal(200, 80)), 1)
        channel = np.random.choice(["mobile", "web", "atm"], p=[0.85, 0.10, 0.05])
        rows.append(create_row(sender, receiver, amount, ts, is_new_recipient, is_new_device, device_count_30d, location_diff_km, channel, True))

    # Build DataFrame and compute sequence features (time gap, velocity, recipient_count, amount_norm, freq_norm)
    df = pd.DataFrame(rows)
    # create realistic timestamps and sort
    df['timestamp'] = [sample_timestamp() for _ in range(len(df))]
    df = df.sort_values('timestamp').reset_index(drop=True)

    # per-user sequential calculations
    last_tx = {}
    recipient_hist = {}  # stores tuples (timestamp, recipient_id)
    user_times = {}

    # We used tx_id as unique but didn't keep recipient id string in row; use tx_id as placeholder for recipient history to mimic counts
    for idx, r in df.iterrows():
        uid = r['user_id']
        ts = r['timestamp']
        # time gap
        if uid in last_tx:
            gap_min = (ts - last_tx[uid]).total_seconds() / 60.0
            df.at[idx, 'time_gap_prev_min'] = int(max(0, gap_min))
        else:
            df.at[idx, 'time_gap_prev_min'] = 999999
        last_tx[uid] = ts

        # recipient_count_30d (we approximate by counting unique tx recipients simulated via tx_id)
        if uid not in recipient_hist:
            recipient_hist[uid] = []
        window_start = ts - pd.Timedelta(days=30)
        recipient_hist[uid] = [t for t in recipient_hist[uid] if t[0] >= window_start]
        recipients_set = set([t[1] for t in recipient_hist[uid]])
        # increment by 1 if this tx_id not in set (we don't have true recipient id stored; but this mimics dynamics)
        df.at[idx, 'recipient_count_30d'] = len(recipients_set) + 1
        recipient_hist[uid].append((ts, r['tx_id']))

        # velocity 1h and 24h
        if uid not in user_times:
            user_times[uid] = []
        user_times[uid].append(ts)
        one_hour = ts - pd.Timedelta(hours=1)
        day_24 = ts - pd.Timedelta(hours=24)
        cnt1 = sum(1 for t in user_times[uid] if t >= one_hour)
        cnt24 = sum(1 for t in user_times[uid] if t >= day_24)
        df.at[idx, 'velocity_1h'] = cnt1
        df.at[idx, 'velocity_24h'] = cnt24

    # amount_norm per user (z-score)
    user_mean = df.groupby('user_id')['amount'].mean().to_dict()
    user_std = df.groupby('user_id')['amount'].std().replace(0, 1).to_dict()
    df['amount_norm'] = df.apply(lambda r: (r['amount'] - user_mean.get(r['user_id'], r['amount'])) / (user_std.get(r['user_id'], 1) if user_std.get(r['user_id'], 1) > 0 else 1), axis=1)
    df['amount_log'] = df['amount'].apply(lambda x: float(np.log(x + 1)))
    df['amount_percentile_system'] = df['amount'].rank(pct=True)
    avg_vel = df.groupby('user_id')['velocity_24h'].median().to_dict()
    df['freq_norm'] = df.apply(lambda r: r['velocity_24h'] / (avg_vel.get(r['user_id'], 1) if avg_vel.get(r['user_id'], 1) > 0 else 1), axis=1)
    df['global_anomaly_score_prev'] = df['is_fraud'].apply(lambda x: float(np.random.uniform(0.6, 1.0)) if x == 1 else float(np.random.uniform(0.0, 0.3)))

    # drop timestamp (features kept)
    df = df.drop(columns=['timestamp'])

    # Reorder columns to the 21 requested + is_fraud label
    cols = ["tx_id","user_id","amount","amount_log","amount_norm","hour_of_day","day_of_week","is_weekend",
            "time_gap_prev_min","velocity_1h","velocity_24h","freq_norm","is_new_recipient","recipient_count_30d",
            "is_new_device","device_count_30d","location_diff_km","channel","account_age_days","amount_percentile_system",
            "global_anomaly_score_prev","is_fraud"]
    df = df[cols]

    # Prepare files for download
    iso_df = df.copy()  # Unsupervised dataset (is_fraud kept only for evaluation)
    lgb_df = df.copy()
    if include_labels_for_lgb:
        lgb_df = lgb_df.rename(columns={"is_fraud": "label"})
    else:
        lgb_df["label"] = 0  # placeholder

    # convert channel to numeric for LightGBM convenience
    lgb_df["channel_code"] = lgb_df["channel"].map({"mobile":0, "web":1, "atm":2})

    # Create CSV buffers
    iso_buffer = io.StringIO()
    lgb_buffer = io.StringIO()
    iso_df.to_csv(iso_buffer, index=False)
    lgb_df.to_csv(lgb_buffer, index=False)

    st.success("Tạo xong — tải xuống bên dưới:")
    st.download_button("Download Isolation CSV (unsupervised)", iso_buffer.getvalue(), file_name="isolation_train.csv", mime="text/csv")
    st.download_button("Download LightGBM CSV (supervised)", lgb_buffer.getvalue(), file_name="lightgbm_train.csv", mime="text/csv")
    st.write("Preview (8 rows):")
    st.dataframe(df.head(8))

st.markdown("---")
st.caption("Ghi chú: dataset được mô phỏng cho mục đích POC/huấn luyện. Để đưa vào production, cần làm sạch, mask PII, và so sánh với dữ liệu thật trước khi triển khai.")
