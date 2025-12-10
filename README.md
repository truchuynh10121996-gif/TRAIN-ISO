# Vietnam Fraud & Scam Detection - Synthetic Data Generator

## Tổng quan

Tool tạo dữ liệu giả lập giao dịch ngân hàng Việt Nam để train các model phát hiện gian lận/lừa đảo.

### Các loại fraud được mô phỏng:

**SCAM (Lừa đảo - Nạn nhân tự chuyển tiền):**
- `impersonation` (35%): Giả danh công an/viện kiểm sát
- `job_scam` (25%): Việc làm online, đầu tư sinh lời
- `lottery_prize` (15%): Trúng thưởng, nhận quà
- `romance_scam` (10%): Lừa tình cảm qua mạng
- `investment_scam` (15%): Đầu tư crypto/forex lừa đảo

**ATO (Account Takeover - Hacker chiếm tài khoản):**
- `credential_theft` (50%): Đánh cắp mật khẩu
- `sim_swap` (30%): Chiếm đoạt SIM
- `phishing` (20%): Website giả mạo

---

## Files xuất ra

| File | Mô tả | Nội dung |
|------|-------|----------|
| `*_full.csv` | Đầy đủ tất cả features để phân tích | 40 columns |
| `*_train.csv` | Training data chung (không có fraud_type) | 39 columns |
| `vietnam_IF_features_*.csv` | **Isolation Forest - để TRAIN** | 14 features (không có label) |
| `vietnam_IF_labels_*.csv` | **Isolation Forest - để EVALUATE** | is_fraud + fraud_type |
| `vietnam_LGBM_train_*.csv` | **LightGBM - để TRAIN** | 29 features + 1 label |

---

## Features cho Isolation Forest

**Tổng: 14 features (KHÔNG có label trong file train)**

> ⚠️ **QUAN TRỌNG**: Isolation Forest là thuật toán **unsupervised** - **KHÔNG dùng label để train!**
> Label (`is_fraud`) được xuất ra file riêng để dùng cho việc **evaluate** model.

Isolation Forest hoạt động bằng cách "cô lập" các điểm bất thường thông qua random splits. Hiệu quả nhất với:
- Features continuous có phân bố rõ ràng
- Anomalies là outliers về mặt giá trị
- Ít features categorical

### Files xuất ra cho Isolation Forest:

| File | Nội dung | Mục đích |
|------|----------|----------|
| `vietnam_IF_features_*.csv` | 14 features | **Train** model |
| `vietnam_IF_labels_*.csv` | is_fraud, fraud_type | **Evaluate** model |

### Danh sách features (14 features):

| # | Feature | Loại | Mô tả | Tại sao quan trọng |
|---|---------|------|-------|-------------------|
| 1 | `amount_log` | Continuous | Log của số tiền giao dịch | Giảm skewness, outlier detection tốt hơn |
| 2 | `amount_deviation_ratio` | Continuous | `amount / median_amount_của_user` | Phát hiện GD bất thường so với hành vi user |
| 3 | `amount_vs_avg_user` | Continuous | `amount / mean_30d_của_user` | Phát hiện GD đột biến |
| 4 | `hours_since_prev_tx` | Continuous | Số giờ kể từ GD trước | Time-based anomaly |
| 5 | `velocity_1h` | Continuous | Số GD trong 1 giờ qua | Burst detection |
| 6 | `velocity_24h` | Continuous | Số GD trong 24 giờ qua | Activity level |
| 7 | `velocity_ratio` | Continuous | `velocity_1h / velocity_24h` | Phát hiện đột biến ngắn hạn |
| 8 | `location_diff_km` | Continuous | Khoảng cách từ vị trí thường | Geographic anomaly |
| 9 | `location_anomaly` | Continuous | `log(1 + location_diff_km)` | Log scale cho outlier detection |
| 10 | `hour_deviation` | Continuous | `|hour - typical_hour_của_user|` | Temporal anomaly |
| 11 | `is_night_hours` | Binary | 1 nếu 22h-5h | High-risk time window |
| 12 | `is_new_recipient` | Binary | 1 nếu người nhận mới | Key fraud indicator |
| 13 | `is_new_device` | Binary | 1 nếu thiết bị mới | ATO indicator |
| 14 | `account_age_risk` | Continuous | `1 / log(account_age_days)` | Tài khoản mới = rủi ro cao hơn |

### Cách sử dụng đúng với Isolation Forest:

```python
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.metrics import classification_report

# 1. Load features (KHÔNG có label)
X = pd.read_csv('vietnam_IF_features_50000_rows.csv')

# 2. Train Isolation Forest (unsupervised - KHÔNG dùng label)
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # ~5% fraud rate
    random_state=42
)
model.fit(X)  # Chỉ có X, không có y!

# 3. Predict (-1 = anomaly, 1 = normal)
predictions = model.predict(X)
predicted_fraud = (predictions == -1).astype(int)

# 4. Evaluate với labels riêng
labels = pd.read_csv('vietnam_IF_labels_50000_rows.csv')
print(classification_report(labels['is_fraud'], predicted_fraud))
```

---

## Features cho LightGBM

**Tổng: 29 features + 1 label = 30 columns**

LightGBM là thuật toán **supervised gradient boosting** - cần label để train. Ưu điểm:
- Xử lý tốt categorical features
- Học được feature interactions phức tạp
- Xử lý được missing values
- Nhanh và hiệu quả với large datasets

### Danh sách features:

| # | Feature | Loại | Mô tả | Tại sao quan trọng |
|---|---------|------|-------|-------------------|
| **Transaction Info** |
| 1 | `transaction_type` | Categorical (0-7) | Loại GD | Transfer (1) có fraud rate cao nhất |
| 2 | `amount_log` | Continuous | Log của số tiền | Amount pattern |
| 3 | `amount_tier` | Ordinal (0-3) | Mức amount | 0: <200k, 1: 200k-2M, 2: 2M-20M, 3: >20M |
| 4 | `amount_vs_avg_user` | Continuous | Tỷ lệ so với trung bình user | Đột biến amount |
| 5 | `channel` | Categorical (0-2) | 0: Mobile, 1: Web, 2: ATM | Web có risk cao hơn |
| 6 | `channel_risk` | Continuous | Risk score theo channel | Pre-encoded risk |
| 7 | `tx_type_risk` | Continuous | Risk score theo loại GD | Pre-encoded risk |
| **Time Features** |
| 8 | `hour_of_day` | Continuous (0-23) | Giờ trong ngày | Night hours rủi ro hơn |
| 9 | `day_of_week` | Categorical (0-6) | Ngày trong tuần | Weekend patterns |
| 10 | `is_weekend` | Binary | 1 nếu thứ 7/CN | Context feature |
| 11 | `is_night_hours` | Binary | 1 nếu 22h-5h | High-risk window |
| 12 | `is_salary_period` | Binary | 1 nếu ngày 25-5 | VN salary cycle |
| 13 | `is_bill_period` | Binary | 1 nếu ngày 1-10 | VN bill payment cycle |
| **Behavioral Features** |
| 14 | `time_gap_prev_min` | Continuous | Phút kể từ GD trước | Rapid succession = suspicious |
| 15 | `velocity_1h` | Continuous | Số GD trong 1 giờ | Burst activity |
| 16 | `velocity_24h` | Continuous | Số GD trong 24 giờ | Daily activity |
| **Recipient/Device Features** |
| 17 | `is_new_recipient` | Binary | 1 nếu người nhận mới | Key fraud indicator |
| 18 | `recipient_count_30d` | Continuous | Số người nhận trong 30 ngày | Diversity indicator |
| 19 | `is_new_device` | Binary | 1 nếu thiết bị mới | ATO indicator |
| 20 | `device_count_30d` | Continuous | Số thiết bị trong 30 ngày | Multi-device risk |
| **Location/Account Features** |
| 21 | `location_diff_km` | Continuous | Khoảng cách từ vị trí thường | Geographic anomaly |
| 22 | `account_age_days` | Continuous | Tuổi tài khoản (ngày) | New account risk |
| **Risk Indicators** |
| 23 | `is_first_large_tx` | Binary | 1 nếu là GD lớn đầu tiên | Scam indicator |
| 24 | `recipient_is_suspicious` | Binary | 1 nếu người nhận đáng ngờ | Blacklist indicator |
| **Engineered Features** |
| 25 | `behavioral_risk_score` | Continuous | Combined risk từ behavior | Weighted sum of behaviors |
| 26 | `time_context_risk` | Continuous | Risk từ weekend + night | Time-based risk |
| 27 | `user_activity_level` | Continuous | Mức hoạt động của user | Activity baseline |
| 28 | `recipient_diversity` | Continuous | `recipient_count_30d / 30` | Normalized diversity |
| - | `is_fraud` | Binary | Label (0/1) | Target variable |

### Cách sử dụng với LightGBM:

```python
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv('vietnam_fraud_lightgbm_50000_rows.csv')

# Tách features và label
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Định nghĩa categorical features
cat_features = ['transaction_type', 'channel', 'day_of_week']

# Tạo LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalance
    'verbose': -1
}

# Train
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(50)]
)

# Predict
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)
print(importance.head(10))
```

---

## Tại sao loại bỏ các features redundant?

### 1. `amount_norm` (đã loại bỏ, giữ `amount_log`)

**Lý do:**
- `amount_norm = (amount - min) / (max - min)` và `amount_log = log(1 + amount)` có correlation cao (~0.85+)
- `amount_log` tốt hơn vì:
  - Giảm skewness của phân bố amount
  - Xử lý outliers tốt hơn
  - Phổ biến trong fraud detection
  - Không bị ảnh hưởng bởi min/max của dataset

### 2. `freq_norm` (đã loại bỏ, giữ `velocity_24h`)

**Lý do:**
- `freq_norm = (velocity_24h - min) / (max - min)` là normalized version của `velocity_24h`
- Correlation = 1.0 (perfect multicollinearity)
- `velocity_24h` tốt hơn vì:
  - Giá trị có ý nghĩa thực tế (số GD/ngày)
  - Dễ interpret và debug
  - Không bị ảnh hưởng bởi min/max của dataset

---

## Tối ưu hóa cho từng model

### Isolation Forest - Tại sao chọn các features này?

1. **Continuous features là chính**: IF hoạt động bằng random splits, continuous features cho phép split tốt hơn categorical
2. **Focus vào outlier detection**: Các features như `amount_deviation_ratio`, `velocity_ratio` được thiết kế để làm nổi bật outliers
3. **Ít features**: IF hiệu quả với số lượng features vừa phải (~10-20), tránh curse of dimensionality
4. **Log transforms**: `amount_log`, `location_anomaly` giúp IF phát hiện outliers ở các scale khác nhau

### LightGBM - Tại sao chọn các features này?

1. **Mix categorical + continuous**: LGBM xử lý tốt cả hai loại
2. **Risk-encoded features**: `tx_type_risk`, `channel_risk`, `behavioral_risk_score` giúp model học nhanh hơn
3. **Feature interactions**: Nhiều features để LGBM có thể học các patterns phức tạp
4. **Context features**: `is_salary_period`, `is_bill_period` là VN-specific patterns

---

## Khuyến nghị sử dụng

### Cho production system:

1. **Dùng cả 2 model kết hợp**:
   - Isolation Forest: Real-time scoring (nhanh, không cần retrain thường xuyên)
   - LightGBM: Batch scoring với accuracy cao hơn

2. **Ensemble approach**:
   ```python
   # Combine scores
   final_score = 0.4 * isolation_score + 0.6 * lgbm_probability
   ```

3. **Threshold tuning**:
   - Với fraud detection, optimize cho Precision-Recall tradeoff
   - Thường dùng threshold thấp hơn 0.5 để catch more frauds

### Recommended dataset size:
- **Minimum**: 10,000 rows (1,000 fraud cases với 10% ratio)
- **Recommended**: 50,000+ rows
- **Production**: 100,000+ rows với lịch sử 6-12 tháng

---

## Changelog

### v2.0.0 (Current)
- Loại bỏ `amount_norm` và `freq_norm` để tránh multicollinearity
- Thêm 6 features mới cho Isolation Forest
- Thêm 8 features mới cho LightGBM
- Thêm download options riêng cho từng model
- Cải thiện UI với tabs giải thích features

### v1.0.0
- Initial release với Vietnamese fraud patterns
- 30 base features
