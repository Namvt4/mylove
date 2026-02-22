"""
Giai đoạn 3B: Mô hình Prophet (Additive Model)
- y(t) = g(t) + s(t) + h(t) + ε
- Regressors: DXY, WTI (biến ngoại sinh)
- Decomposition: trend, seasonality
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "output", "figures")


def prepare_data(df_prophet, test_ratio=0.2):
    """Chia dữ liệu Prophet thành train/test."""
    split_idx = int(len(df_prophet) * (1 - test_ratio))

    train = df_prophet.iloc[:split_idx].copy()
    test = df_prophet.iloc[split_idx:].copy()

    print(f"\n📊 Prophet Train/Test Split:")
    print(f"   Train: {len(train)} mẫu ({train['ds'].iloc[0].date()} → {train['ds'].iloc[-1].date()})")
    print(f"   Test:  {len(test)} mẫu ({test['ds'].iloc[0].date()} → {test['ds'].iloc[-1].date()})")

    return train, test


def build_prophet_model(train, test):
    """Xây dựng mô hình Prophet với regressors."""
    print("\n🔮 Xây dựng mô hình Prophet...")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode="multiplicative",
    )

    # Thêm regressors ngoại sinh
    if "WTI" in train.columns:
        model.add_regressor("WTI", mode="multiplicative")
        print("   ➕ Regressor: WTI (Crude Oil)")
    if "DXY" in train.columns:
        model.add_regressor("DXY", mode="multiplicative")
        print("   ➕ Regressor: DXY (USD Index)")

    # Fit mô hình
    model.fit(train)
    print("   ✅ Mô hình đã được huấn luyện")

    # Predict trên test set
    forecast_test = model.predict(test)
    forecast_train = model.predict(train)

    return model, forecast_train, forecast_test


def plot_decomposition(model, forecast_train, train):
    """Vẽ biểu đồ decomposition của Prophet (trend, seasonality)."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Component plot
    fig = model.plot_components(forecast_train)
    fig.suptitle("Prophet Decomposition: Giá Vàng", fontsize=14, fontweight="bold", y=1.02)
    path = os.path.join(FIGURES_DIR, "prophet_decomposition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   💾 Decomposition chart: {path}")

    # Forecast plot
    fig2 = model.plot(forecast_train)
    ax = fig2.gca()
    ax.set_title("Prophet Forecast - Training Period", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Giá Vàng (USD)")
    path2 = os.path.join(FIGURES_DIR, "prophet_forecast.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"   💾 Forecast chart: {path2}")

    return path, path2


def run(df_prophet):
    """Pipeline Prophet hoàn chỉnh."""
    print("\n" + "=" * 60)
    print("🔮 GIAI ĐOẠN 3B: MÔ HÌNH PROPHET")
    print("=" * 60)

    train, test = prepare_data(df_prophet)

    model, forecast_train, forecast_test = build_prophet_model(train, test)

    plot_decomposition(model, forecast_train, train)

    # Lấy predictions
    y_train_actual = train["y"].values
    y_test_actual = test["y"].values
    y_pred_train = forecast_train["yhat"].values
    y_pred_test = forecast_test["yhat"].values

    return {
        "model": model,
        "y_train": y_train_actual,
        "y_test": y_test_actual,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "forecast_train": forecast_train,
        "forecast_test": forecast_test,
        "test_dates": test["ds"].values,
        "train_dates": train["ds"].values,
    }


if __name__ == "__main__":
    import data_collection
    _, df_prophet = data_collection.run()
    run(df_prophet)
