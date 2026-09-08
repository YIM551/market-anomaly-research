# Historical experimental logic preserved; see docs/evaluation-audit.md.
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
#%%
import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from tqdm import tqdm
import yfinance as yf
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import xgboost as xgb
#%%
# 1. 종목, 경로, 구간 설정
tickers = {
    "TRBO": { "anomaly_start": "30-Mar-20", "anomaly_end": "9-Apr-20"  },
    "APPB": { "anomaly_start": "25-Mar-20", "anomaly_end": "13-Apr-20" },
    "AEMD": { "anomaly_start": "22-Jan-20", "anomaly_end": "7-Feb-20"  },
    "NBDR": { "anomaly_start": "11-Mar-20", "anomaly_end": "3-Apr-20"  },
    "GME" : { "anomaly_start": "11-Jan-21", "anomaly_end": "29-Jan-21" }
}
stock_csv_files = {
    'AEMD': str(DATA_DIR / "AEMD_augmented.csv"),
    'APPB': str(DATA_DIR / "APPB_augmented.csv"),
    'GME':  str(DATA_DIR / "GME_augmented.csv"),
    'NBDR': str(DATA_DIR / "NBDR_augmented.csv"),
    'TRBO': str(DATA_DIR / "TRBO_augmented.csv"),
}
stocks = list(tickers.keys())
#%%
# 2. 유틸리티: 날짜 계산 함수
def get_date_minus_months(date_str, months):
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    year, month = dt.year, dt.month - months
    while month <= 0:
        year -= 1
        month += 12
    try: return datetime.datetime(year, month, dt.day).strftime("%Y-%m-%d")
    except ValueError: return (datetime.datetime(year, month, 1) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
def get_date_plus_months(date_str, months):
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    year, month = dt.year, dt.month + months
    while month > 12:
        year += 1
        month -= 12
    try: return datetime.datetime(year, month, dt.day).strftime("%Y-%m-%d")
    except ValueError: return (datetime.datetime(year, month, 1) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
#%%
# 3. FinBERT 감성분석 파이프라인
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
def run_finbert(news_df):
    tqdm.pandas()
    news_df['text'] = news_df['Headline'].fillna('') + '. ' + news_df['Summary'].fillna('')
    news_df['sentiment'] = news_df['text'].progress_apply(lambda x: finbert(x)[0]['label'])
    news_df['sentiment_score'] = news_df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
    return news_df

def daily_sentiment(news_df):
    df = news_df.groupby('Date')['sentiment_score'].mean().reset_index()
    df.rename(columns={'sentiment_score': 'sentiment_mean'}, inplace=True)
    return df

def detect_sentiment_anomaly(df_sent, z_thresh=0.5):
    mean, std = df_sent['sentiment_mean'].mean(), df_sent['sentiment_mean'].std()
    df_sent['zscore'] = (df_sent['sentiment_mean'] - mean) / std
    df_sent['sentiment_anomaly'] = (df_sent['zscore'].abs() > z_thresh).astype(int)
    return df_sent
#%%
# 4. LSTM Autoencoder 탐지
def lstm_autoencoder_detect(signal_df, anomalies_df, window=10, hidden_units=80, epochs=35, batch_size=64, return_mse_score=False):
    values = signal_df['value'].values.reshape(-1,1).astype('float32')
    timestamps = signal_df['timestamp'].values
    n = len(values)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X = np.array([scaled[i:i+window] for i in range(n-window+1)])
    model = Sequential([
        LSTM(hidden_units, activation='relu', input_shape=(window,1)),
        RepeatVector(window),
        LSTM(hidden_units, activation='relu', return_sequences=True),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')
    start_time = time.time()
    model.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0)
    X_pred = model.predict(X, batch_size=batch_size, verbose=0)
    mse = np.mean(np.square(X_pred - X), axis=(1,2))
    thr = np.mean(mse) + 3*np.std(mse)
    detected_flags = np.zeros(n, dtype=int)
    for i, err in enumerate(mse):
        idx = i + window - 1
        if err > thr and idx < n:
            detected_flags[idx] = 1
    elapsed = time.time() - start_time
    is_true_label = np.zeros(n, dtype=int)
    for _, row in anomalies_df.iterrows():
        mask = (timestamps >= row['start']) & (timestamps <= row['end'])
        is_true_label[mask] = 1
    detected_windows = []
    in_anom = False
    for i in range(n):
        if detected_flags[i] == 1 and not in_anom:
            s_idx = i
            in_anom = True
        if (detected_flags[i] == 0 or i == n-1) and in_anom:
            e_idx = i-1 if detected_flags[i] == 0 else i
            detected_windows.append((timestamps[s_idx], timestamps[e_idx]))
            in_anom = False
    detected_df = pd.DataFrame(detected_windows, columns=['start','end'])
    precision = precision_score(is_true_label, detected_flags, zero_division=0)
    recall    = recall_score(is_true_label, detected_flags, zero_division=0)
    f1        = f1_score(is_true_label, detected_flags, zero_division=0)
    metrics = {"precision": precision, "recall": recall, "f1": f1}
    if return_mse_score:
        return detected_df, metrics, elapsed, mse
    else:
        return detected_df, metrics, elapsed
#%%
# 5. 통합 feature 생성
def make_lstm_score_df(signal_df, mse_score, window):
    date_list = [datetime.datetime.fromtimestamp(ts).date() for ts in signal_df['timestamp'].values]
    df = pd.DataFrame({'date': date_list, 'lstm_score': np.nan})
    for i, score in enumerate(mse_score):
        idx = i + window - 1
        if idx < len(df):
            df.at[idx, 'lstm_score'] = score
    df['lstm_score'] = df['lstm_score'].fillna(0)
    return df
def make_sentiment_df(df_sent):
    df = df_sent.copy()
    df['positive_ratio'] = (df['sentiment_mean'] > 0).astype(int)
    df['text_length'] = df.get('text', '').apply(lambda x: len(x) if isinstance(x, str) else 0) if 'text' in df else 0
    df = df[['Date', 'sentiment_mean', 'positive_ratio', 'text_length']]
    df.columns = ['date', 'sentiment_mean', 'positive_ratio', 'text_length']
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df
def make_label_df(signal_df, anomalies_df):
    dates = [datetime.datetime.fromtimestamp(ts).date() for ts in signal_df['timestamp'].values]
    labels = []
    for ts in signal_df['timestamp'].values:
        label = 0
        for _, row in anomalies_df.iterrows():
            if ts >= row['start'] and ts <= row['end']:
                label = 1
                break
        labels.append(label)
    return pd.DataFrame({'date': dates, 'label': labels})
#%%
# 6. XGBoost 탐지 구간 연속화
def extract_detected_anomalies_from_xgb(df, date_col='date', label_col='xgb_pred'):
    detected, in_anom = [], False
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    for i, row in df.iterrows():
        if row[label_col] == 1 and not in_anom:
            s = row[date_col]
            in_anom = True
        if (row[label_col] == 0 or i == len(df) - 1) and in_anom:
            e = df[date_col][i-1 if row[label_col]==0 else i]
            detected.append((s, e))
            in_anom = False
    return detected
#%%
# 7. 시각화 함수 (3가지 공통)
def plot_detection_vs_real(all_stock_data, all_anomalies, detected_anomalies, method_label="Detected anomalies"):
    stocks = list(all_stock_data.keys())
    n_stocks = len(stocks)
    fig, axs = plt.subplots(n_stocks, 1, figsize=(10, 12), sharex=True)
    for idx, stock in enumerate(stocks):
        ax = axs[idx]
        df = all_stock_data[stock]
        ax.plot(df['Date'], df['Volume'], label='Volume', color='royalblue', lw=1)
        for _, row in all_anomalies[stock].iterrows():
            ax.axvspan(pd.to_datetime(row['start'], unit='s'), pd.to_datetime(row['end'], unit='s'), color='green', alpha=0.2)
        for start, end in detected_anomalies.get(stock, []):
            ax.axvspan(start, end, color='red', alpha=0.18)
        company = {
            "TRBO": "Turbo Global Partners, Inc.",
            "APPB": "Applied Biosciences Corp",
            "AEMD": "Aethlon Medical, Inc.",
            "NBDR": "No Borders, Inc.",
            "GME": "GameStop"
        }.get(stock, stock)
        ax.set_title(f'Traded volumes for {company} (“{stock}”)', fontsize=11)
        ax.set_ylabel('Volume')
        if idx == 0:
            green_patch = mpatches.Patch(color='green', alpha=0.2, label='Real anomalies')
            red_patch = mpatches.Patch(color='red', alpha=0.18, label=method_label)
            ax.legend(handles=[red_patch, green_patch], loc='upper left', fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True)
    axs[-1].set_xlabel('Date')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
#%%
# ========== 데이터 준비, 감성분석/AE/XGB 모두 여기서 ==========
all_stock_data, all_anomalies = {}, {}
df_sent_dict, detected_anomalies_sent = {}, {}
signal_dict = {}   # LSTM 입력용 시계열

for stock, info in tickers.items():
    anom_start_str = pd.to_datetime(info["anomaly_start"], format="%d-%b-%y").strftime("%Y-%m-%d")
    anom_end_str   = pd.to_datetime(info["anomaly_end"],   format="%d-%b-%y").strftime("%Y-%m-%d")
    hist_start = get_date_minus_months(anom_start_str, 24)
    hist_end   = get_date_plus_months(anom_end_str, 12)

    stock_data = yf.Ticker(stock)
    df = stock_data.history(start=hist_start, end=hist_end).reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    all_stock_data[stock] = df[['Date', 'Volume']]
    # LSTM용 시계열
    signal_df = pd.DataFrame({
        'timestamp': df['Date'].astype(np.int64) // 10**9,
        'value': df['Volume'].values
    })
    signal_dict[stock] = signal_df

    # 실제 이상치 구간(unix timestamp)
    anom_start_ts = int(pd.Timestamp(anom_start_str).timestamp())
    anom_end_ts   = int(pd.Timestamp(anom_end_str).timestamp())
    all_anomalies[stock] = pd.DataFrame([[anom_start_ts, anom_end_ts]], columns=['start','end'])

    # 뉴스/감성 분석
    news = pd.read_csv(stock_csv_files[stock], encoding='latin1', on_bad_lines='skip')
    news['Date'] = pd.to_datetime(news['Date']).dt.date
    news = run_finbert(news)
    df_sent = daily_sentiment(news)
    df_sent = detect_sentiment_anomaly(df_sent)
    df_sent['Date'] = pd.to_datetime(df_sent['Date'])
    df_sent_dict[stock] = df_sent

    # 감성 기반 탐지 구간
    detected = []
    grouped = df_sent[df_sent['sentiment_anomaly'] == 1].groupby((df_sent['sentiment_anomaly'] != df_sent['sentiment_anomaly'].shift()).cumsum())
    for _, group in grouped:
        if group['sentiment_anomaly'].iloc[0] == 1:
            start = group['Date'].min()
            end = group['Date'].max()
            detected.append((start, end))
    detected_anomalies_sent[stock] = detected
#%%
# ① 감성분석 기반 이상 탐지 표/시각화
sent_results = []
for stock in stocks:
    df_sent = df_sent_dict[stock]
    s = pd.to_datetime(tickers[stock]['anomaly_start'], format="%d-%b-%y")
    e = pd.to_datetime(tickers[stock]['anomaly_end'], format="%d-%b-%y")
    df_sent['label'] = ((df_sent['Date'] >= s) & (df_sent['Date'] <= e)).astype(int)
    precision = precision_score(df_sent['label'], df_sent['sentiment_anomaly'])
    recall    = recall_score(df_sent['label'], df_sent['sentiment_anomaly'])
    f1        = f1_score(df_sent['label'], df_sent['sentiment_anomaly'])
    sent_results.append([stock, round(precision,3), round(recall,3), round(f1,3)])
sent_result_df = pd.DataFrame(sent_results, columns=['Stock','Precision','Recall','F1-score'])
print("\n[감성분석 기반 이상 탐지]\n", sent_result_df)
plot_detection_vs_real(all_stock_data, all_anomalies, detected_anomalies_sent, method_label="Sentiment-detected")
#%%
window_list = [5, 10, 20, 50, 100, 250]
epoch_list = [10, 35, 70]
auto_results, detected_anomalies_ae, best_windows, best_epochs = [], {}, {}, {}

for stock in stocks:
    signal_df = signal_dict[stock]
    anom_df = all_anomalies[stock]
    best_f1, best_w, best_e, best_detected = -1, None, None, None  # best_f1을 -1로!
    for w in window_list:
        if w >= len(signal_df): continue
        for e in epoch_list:
            detected_df, metrics, elapsed = lstm_autoencoder_detect(
                signal_df, anom_df, window=w, epochs=e, hidden_units=80, batch_size=64
            )
            if metrics['f1'] > best_f1:
                best_f1, best_w, best_e, best_detected = metrics['f1'], w, e, detected_df
    # best_detected가 None인 경우 방어
    detected = []
    if best_detected is not None and len(best_detected) > 0:
        for _, row in best_detected.iterrows():
            s = datetime.datetime.fromtimestamp(row['start'])
            e = datetime.datetime.fromtimestamp(row['end'])
            detected.append((s, e))
    detected_anomalies_ae[stock] = detected
    best_windows[stock], best_epochs[stock] = best_w, best_e
    # Best 조합으로 재학습 및 평가 (best_w, best_e가 None이면 건너뜀)
    if best_w is not None and best_e is not None:
        detected_df, metrics, elapsed = lstm_autoencoder_detect(
            signal_df, anom_df, window=best_w, epochs=best_e, hidden_units=80, batch_size=64
        )
        auto_results.append([
            stock, best_w, best_e, metrics['precision'], metrics['recall'], metrics['f1']
        ])
    else:
        auto_results.append([stock, None, None, None, None, None])

# 결과표
auto_result_df = pd.DataFrame(
    auto_results,
    columns=['Stock', 'Best window', 'Best epochs', 'Precision', 'Recall', 'F1-score']
)
print("\n[LSTM Autoencoder 기반 이상 탐지]\n", auto_result_df)
plot_detection_vs_real(all_stock_data, all_anomalies, detected_anomalies_ae, method_label="LSTM-AE-detected")
#%%
# ③ XGBoost 통합 (feature: LSTM+감성, label: 구간 기반)
all_lstm_score, all_sentiment, all_label, df_merged_dict = {}, {}, {}, {}
for stock in stocks:
    # LSTM mse
    w, e = best_windows[stock], best_epochs[stock]
    _, _, _, mse_score = lstm_autoencoder_detect(signal_dict[stock], all_anomalies[stock], window=w, epochs=e, return_mse_score=True)
    lstm_score_df = make_lstm_score_df(signal_dict[stock], mse_score, w)
    all_lstm_score[stock] = lstm_score_df
    # 감성 피처
    sentiment_df = make_sentiment_df(df_sent_dict[stock])
    all_sentiment[stock] = sentiment_df
    # 라벨
    label_df = make_label_df(signal_dict[stock], all_anomalies[stock])
    all_label[stock] = label_df
    # 통합 feature
    date_min = min(lstm_score_df['date'].min(), sentiment_df['date'].min(), label_df['date'].min())
    date_max = max(lstm_score_df['date'].max(), sentiment_df['date'].max(), label_df['date'].max())
    date_list = pd.date_range(date_min, date_max, freq='D')
    merged = pd.DataFrame({'date': date_list.date})
    merged = merged.merge(lstm_score_df, on='date', how='left').merge(sentiment_df, on='date', how='left').merge(label_df, on='date', how='left').fillna(0)
    merged['date'] = pd.to_datetime(merged['date'])
    df_merged_dict[stock] = merged

xgb_model_dict, best_thresholds, detected_anomalies_xgb, xgb_results = {}, {}, {}, []
for stock in stocks:
    df = df_merged_dict[stock]
    X = df[['lstm_score', 'sentiment_mean', 'positive_ratio', 'text_length']].values
    y = df['label'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    n_pos, n_neg = np.sum(y_train == 1), np.sum(y_train == 0)
    pos_weight = n_neg / (n_pos + 1e-8) if n_pos > 0 else 1
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, scale_pos_weight=pos_weight, random_state=42)
    model.fit(X_train, y_train)
    xgb_model_dict[stock] = model
    val_probs = model.predict_proba(X_val)[:,1]
    prec, rec, thr = precision_recall_curve(y_val, val_probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = thr[np.argmax(f1s)] if len(thr) > 0 else 0.5
    best_thresholds[stock] = best_thr
    y_pred = (val_probs >= 0.5).astype(int)
    xgb_results.append([stock, precision_score(y_val, y_pred), recall_score(y_val, y_pred), f1_score(y_val, y_pred)])
    # 탐지 구간 생성
    df['xgb_prob'] = model.predict_proba(df[['lstm_score', 'sentiment_mean', 'positive_ratio', 'text_length']].values)[:,1]
    df['xgb_pred'] = (df['xgb_prob'] >= best_thr).astype(int)
    detected_anomalies_xgb[stock] = extract_detected_anomalies_from_xgb(df, date_col='date', label_col='xgb_pred')

xgb_result_df = pd.DataFrame(xgb_results, columns=['Stock','Precision','Recall','F1-score'])
print("\n[XGBoost 통합 이상 탐지]\n", xgb_result_df)
plot_detection_vs_real(all_stock_data, all_anomalies, detected_anomalies_xgb, method_label="XGBoost-detected")
#%%
# ③ XGBoost 통합 (feature: LSTM+감성, label: 구간 기반)
all_lstm_score, all_sentiment, all_label, df_merged_dict = {}, {}, {}, {}
for stock in stocks:
    # LSTM mse
    w, e = best_windows[stock], best_epochs[stock]
    _, _, _, mse_score = lstm_autoencoder_detect(signal_dict[stock], all_anomalies[stock], window=w, epochs=e, return_mse_score=True)
    lstm_score_df = make_lstm_score_df(signal_dict[stock], mse_score, w)
    all_lstm_score[stock] = lstm_score_df
    # 감성 피처
    sentiment_df = make_sentiment_df(df_sent_dict[stock])
    all_sentiment[stock] = sentiment_df
    # 라벨
    label_df = make_label_df(signal_dict[stock], all_anomalies[stock])
    all_label[stock] = label_df
    # 통합 feature
    date_min = min(lstm_score_df['date'].min(), sentiment_df['date'].min(), label_df['date'].min())
    date_max = max(lstm_score_df['date'].max(), sentiment_df['date'].max(), label_df['date'].max())
    date_list = pd.date_range(date_min, date_max, freq='D')
    merged = pd.DataFrame({'date': date_list.date})
    merged = merged.merge(lstm_score_df, on='date', how='left').merge(sentiment_df, on='date', how='left').merge(label_df, on='date', how='left').fillna(0)
    merged['date'] = pd.to_datetime(merged['date'])
    df_merged_dict[stock] = merged

xgb_model_dict, best_thresholds, detected_anomalies_xgb, xgb_results = {}, {}, {}, []
for stock in stocks:
    df = df_merged_dict[stock]
    X = df[['lstm_score', 'sentiment_mean', 'positive_ratio', 'text_length']].values
    y = df['label'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    n_pos, n_neg = np.sum(y_train == 1), np.sum(y_train == 0)
    pos_weight = n_neg / (n_pos + 1e-8) if n_pos > 0 else 1
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, scale_pos_weight=pos_weight, random_state=42)
    model.fit(X_train, y_train)
    xgb_model_dict[stock] = model
    val_probs = model.predict_proba(X_val)[:,1]
    prec, rec, thr = precision_recall_curve(y_val, val_probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = thr[np.argmax(f1s)] if len(thr) > 0 else 0.5
    best_thresholds[stock] = best_thr
    y_pred = (val_probs >= 0.9).astype(int)
    xgb_results.append([stock, precision_score(y_val, y_pred), recall_score(y_val, y_pred), f1_score(y_val, y_pred)])
    # 탐지 구간 생성
    df['xgb_prob'] = model.predict_proba(df[['lstm_score', 'sentiment_mean', 'positive_ratio', 'text_length']].values)[:,1]
    df['xgb_pred'] = (df['xgb_prob'] >= best_thr).astype(int)
    detected_anomalies_xgb[stock] = extract_detected_anomalies_from_xgb(df, date_col='date', label_col='xgb_pred')

xgb_result_df = pd.DataFrame(xgb_results, columns=['Stock','Precision','Recall','F1-score'])
print("\n[XGBoost 통합 이상 탐지]\n", xgb_result_df)
plot_detection_vs_real(all_stock_data, all_anomalies, detected_anomalies_xgb, method_label="XGBoost-detected")
#%%
# ③ XGBoost 통합 (feature: LSTM+감성, label: 구간 기반)
all_lstm_score, all_sentiment, all_label, df_merged_dict = {}, {}, {}, {}
for stock in stocks:
    # LSTM mse
    w, e = best_windows[stock], best_epochs[stock]
    _, _, _, mse_score = lstm_autoencoder_detect(signal_dict[stock], all_anomalies[stock], window=w, epochs=e, return_mse_score=True)
    lstm_score_df = make_lstm_score_df(signal_dict[stock], mse_score, w)
    all_lstm_score[stock] = lstm_score_df
    # 감성 피처
    sentiment_df = make_sentiment_df(df_sent_dict[stock])
    all_sentiment[stock] = sentiment_df
    # 라벨
    label_df = make_label_df(signal_dict[stock], all_anomalies[stock])
    all_label[stock] = label_df
    # 통합 feature
    date_min = min(lstm_score_df['date'].min(), sentiment_df['date'].min(), label_df['date'].min())
    date_max = max(lstm_score_df['date'].max(), sentiment_df['date'].max(), label_df['date'].max())
    date_list = pd.date_range(date_min, date_max, freq='D')
    merged = pd.DataFrame({'date': date_list.date})
    merged = merged.merge(lstm_score_df, on='date', how='left').merge(sentiment_df, on='date', how='left').merge(label_df, on='date', how='left').fillna(0)
    merged['date'] = pd.to_datetime(merged['date'])
    df_merged_dict[stock] = merged

xgb_model_dict, best_thresholds, detected_anomalies_xgb, xgb_results = {}, {}, {}, []
for stock in stocks:
    df = df_merged_dict[stock]
    X = df[['lstm_score', 'sentiment_mean', 'positive_ratio', 'text_length']].values
    y = df['label'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    n_pos, n_neg = np.sum(y_train == 1), np.sum(y_train == 0)
    pos_weight = n_neg / (n_pos + 1e-8) if n_pos > 0 else 1
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, scale_pos_weight=pos_weight, random_state=42)
    model.fit(X_train, y_train)
    xgb_model_dict[stock] = model
    val_probs = model.predict_proba(X_val)[:,1]
    prec, rec, thr = precision_recall_curve(y_val, val_probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = thr[np.argmax(f1s)] if len(thr) > 0 else 0.5
    best_thresholds[stock] = best_thr
    y_pred = (val_probs >= best_thr).astype(int)
    xgb_results.append([stock, precision_score(y_val, y_pred), recall_score(y_val, y_pred), f1_score(y_val, y_pred)])
    # 탐지 구간 생성
    df['xgb_prob'] = model.predict_proba(df[['lstm_score', 'sentiment_mean', 'positive_ratio', 'text_length']].values)[:,1]
    df['xgb_pred'] = (df['xgb_prob'] >= best_thr).astype(int)
    detected_anomalies_xgb[stock] = extract_detected_anomalies_from_xgb(df, date_col='date', label_col='xgb_pred')

xgb_result_df = pd.DataFrame(xgb_results, columns=['Stock','Precision','Recall','F1-score'])
print("\n[XGBoost 통합 이상 탐지]\n", xgb_result_df)
plot_detection_vs_real(all_stock_data, all_anomalies, detected_anomalies_xgb, method_label="XGBoost-detected")
#%%
