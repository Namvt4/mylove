"""
Giai đoạn 3A: Mô hình XGBoost (Gradient Boosting)
- Feature engineering: lag features, rolling statistics, returns
- Bayesian Optimization với Optuna
- Early stopping
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import os
import warnings
import json

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def create_features(merged, target="Gold", lags=[1, 2, 3, 5, 7, 10, 14, 21]):
    """Tạo features cho XGBoost từ dữ liệu liên thị trường."""
    df = merged.copy()

    # --- Lag features cho tất cả các biến ---
    for col in ["Gold", "WTI", "DXY"]:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # --- Rolling statistics cho Gold ---
    for window in [5, 10, 21]:
        df[f"Gold_MA_{window}"] = df["Gold"].rolling(window).mean()
        df[f"Gold_STD_{window}"] = df["Gold"].rolling(window).std()
        df[f"Gold_MIN_{window}"] = df["Gold"].rolling(window).min()
        df[f"Gold_MAX_{window}"] = df["Gold"].rolling(window).max()

    # --- Returns (tỷ suất sinh lợi) ---
    for col in ["Gold", "WTI", "DXY"]:
        df[f"{col}_return_1d"] = df[col].pct_change(1)
        df[f"{col}_return_5d"] = df[col].pct_change(5)

    # --- Ratio features ---
    df["Gold_WTI_ratio"] = df["Gold"] / df["WTI"]
    df["Gold_DXY_ratio"] = df["Gold"] / df["DXY"]

    # --- Rolling correlation ---
    df["Gold_DXY_corr_21"] = df["Gold"].rolling(21).corr(df["DXY"])
    df["Gold_WTI_corr_21"] = df["Gold"].rolling(21).corr(df["WTI"])

    # Thêm features ngày
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter

    # Loại bỏ NaN
    df = df.dropna()

    # Tách features và target
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    return X, y, feature_cols


def train_test_split_temporal(X, y, test_ratio=0.2):
    """Chia dữ liệu theo thời gian (không shuffle)."""
    split_idx = int(len(X) * (1 - test_ratio))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(X_train)} mẫu ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"   Test:  {len(X_test)} mẫu ({X_test.index[0].date()} → {X_test.index[-1].date()})")

    return X_train, X_test, y_train, y_test


def optimize_with_optuna(X_train, y_train, n_trials=50):
    """Bayesian Optimization với Optuna để tìm tham số tối ưu."""
    print(f"\n🔧 Optuna Bayesian Optimization ({n_trials} trials)...")

    # Chia validation set từ training set
    val_split = int(len(X_train) * 0.8)
    X_tr = X_train.iloc[:val_split]
    X_val = X_train.iloc[val_split:]
    y_tr = y_train.iloc[:val_split]
    y_val = y_train.iloc[val_split:]

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = XGBRegressor(
            **params,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    study = optuna.create_study(direction="minimize", study_name="XGBoost_Gold")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_rmse = study.best_value

    print(f"\n   ✅ Best RMSE (validation): {best_rmse:.4f}")
    print(f"   📋 Best Parameters:")
    for k, v in best_params.items():
        print(f"      {k}: {v}")

    return best_params, study


def train_final_model(X_train, X_test, y_train, y_test, best_params):
    """Train mô hình cuối cùng với tham số tối ưu."""
    print("\n🏋️ Training mô hình XGBoost cuối cùng...")

    model = XGBRegressor(
        **best_params,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return model, y_pred_train, y_pred_test


def get_feature_importance(model, feature_cols, top_n=15):
    """Lấy feature importance từ XGBoost."""
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importance,
    }).sort_values("Importance", ascending=False)

    print(f"\n📊 Top {top_n} Features quan trọng nhất:")
    for i, row in feat_imp.head(top_n).iterrows():
        bar = "█" * int(row["Importance"] * 100)
        print(f"   {row['Feature']:30s} {row['Importance']:.4f} {bar}")

    return feat_imp


def run(merged):
    """Pipeline XGBoost hoàn chỉnh."""
    print("\n" + "=" * 60)
    print("🤖 GIAI ĐOẠN 3A: MÔ HÌNH XGBOOST + OPTUNA")
    print("=" * 60)

    # Tạo features
    X, y, feature_cols = create_features(merged)

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y)

    # Tối ưu tham số
    best_params, study = optimize_with_optuna(X_train, y_train, n_trials=50)

    # Train mô hình cuối
    model, y_pred_train, y_pred_test = train_final_model(
        X_train, X_test, y_train, y_test, best_params
    )

    # Feature importance
    feat_imp = get_feature_importance(model, feature_cols)

    # Lưu best params
    params_path = os.path.join(OUTPUT_DIR, "xgboost_best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    return {
        "model": model,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "feature_importance": feat_imp,
        "best_params": best_params,
        "test_index": X_test.index,
        "train_index": X_train.index,
    }


if __name__ == "__main__":
    import data_collection
    merged, _ = data_collection.run()
    run(merged)
