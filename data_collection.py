"""
Giai đoạn 1: Thu thập và Tiền xử lý dữ liệu
- Nguồn: Yahoo Finance (yfinance)
- Tickers: Gold (GC=F), Crude Oil (CL=F), USD Index (DX-Y.NYB)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def fetch_data(start="2020-01-01", end="2025-12-31"):
    """Tải dữ liệu từ Yahoo Finance cho Gold, WTI Oil, USD Index."""
    tickers = {
        "Gold": "GC=F",
        "WTI": "CL=F",
        "DXY": "DX-Y.NYB",
    }

    frames = {}
    for name, ticker in tickers.items():
        print(f"  ⏳ Đang tải dữ liệu {name} ({ticker})...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].rename(columns={"Close": name})
        frames[name] = df
        print(f"  ✅ {name}: {len(df)} bản ghi")

    return frames


def preprocess(frames):
    """Merge dữ liệu, xử lý missing values."""
    # Merge tất cả theo ngày
    merged = pd.concat(frames.values(), axis=1, join="inner")
    merged.index = pd.to_datetime(merged.index)
    merged = merged.sort_index()

    # Xử lý missing values bằng forward-fill rồi backward-fill
    missing_before = merged.isnull().sum().sum()
    merged = merged.ffill().bfill()
    missing_after = merged.isnull().sum().sum()

    print(f"\n📊 Dữ liệu sau merge: {len(merged)} bản ghi")
    print(f"   Missing values: {missing_before} → {missing_after}")
    print(f"   Khoảng thời gian: {merged.index[0].date()} → {merged.index[-1].date()}")

    return merged


def prepare_prophet_format(merged, target="Gold"):
    """Chuyển đổi dữ liệu về format Prophet (ds, y) với regressors."""
    df_prophet = merged.reset_index()
    df_prophet = df_prophet.rename(columns={"Date": "ds", target: "y"})

    # Giữ lại regressors
    cols_keep = ["ds", "y"]
    for col in ["WTI", "DXY"]:
        if col in df_prophet.columns:
            cols_keep.append(col)

    df_prophet = df_prophet[cols_keep]
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    return df_prophet


def save_data(merged, df_prophet):
    """Lưu dữ liệu đã xử lý."""
    os.makedirs(DATA_DIR, exist_ok=True)

    merged_path = os.path.join(DATA_DIR, "merged_data.csv")
    prophet_path = os.path.join(DATA_DIR, "prophet_data.csv")

    merged.to_csv(merged_path)
    df_prophet.to_csv(prophet_path, index=False)

    print(f"\n💾 Đã lưu dữ liệu:")
    print(f"   - {merged_path}")
    print(f"   - {prophet_path}")

    return merged_path, prophet_path


def run():
    """Pipeline thu thập & tiền xử lý dữ liệu."""
    print("=" * 60)
    print("📥 GIAI ĐOẠN 1: THU THẬP & TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    frames = fetch_data()
    merged = preprocess(frames)
    df_prophet = prepare_prophet_format(merged)
    save_data(merged, df_prophet)

    return merged, df_prophet


if __name__ == "__main__":
    run()
