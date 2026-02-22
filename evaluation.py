"""
Giai đoạn 4: Đánh giá và So sánh Mô hình
- Metrics: MAE, RMSE, MAPE
- So sánh XGBoost vs Prophet
"""

import numpy as np
import pandas as pd
import os


def calculate_metrics(y_actual, y_predicted, model_name="Model"):
    """Tính MAE, RMSE, MAPE cho một mô hình."""
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    mae = np.mean(np.abs(y_actual - y_predicted))
    rmse = np.sqrt(np.mean((y_actual - y_predicted) ** 2))

    # MAPE - tránh chia cho 0
    mask = y_actual != 0
    mape = np.mean(np.abs((y_actual[mask] - y_predicted[mask]) / y_actual[mask])) * 100

    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 4),
    }


def compare_models(xgb_results, prophet_results):
    """So sánh XGBoost vs Prophet."""
    print("\n" + "=" * 60)
    print("📊 GIAI ĐOẠN 4: ĐÁNH GIÁ & SO SÁNH MÔ HÌNH")
    print("=" * 60)

    # --- Train set metrics ---
    xgb_train = calculate_metrics(
        xgb_results["y_train"], xgb_results["y_pred_train"], "XGBoost (Train)"
    )
    prophet_train = calculate_metrics(
        prophet_results["y_train"], prophet_results["y_pred_train"], "Prophet (Train)"
    )

    # --- Test set metrics ---
    xgb_test = calculate_metrics(
        xgb_results["y_test"], xgb_results["y_pred_test"], "XGBoost (Test)"
    )
    prophet_test = calculate_metrics(
        prophet_results["y_test"], prophet_results["y_pred_test"], "Prophet (Test)"
    )

    all_metrics = [xgb_train, prophet_train, xgb_test, prophet_test]
    df_metrics = pd.DataFrame(all_metrics)

    print("\n📋 Bảng so sánh chi tiết:")
    print("-" * 60)
    print(df_metrics.to_string(index=False))
    print("-" * 60)

    # Xác định winner
    print("\n🏆 Kết luận trên Test Set:")
    test_metrics = [xgb_test, prophet_test]

    for metric in ["MAE", "RMSE", "MAPE (%)"]:
        values = [(m["Model"], m[metric]) for m in test_metrics]
        winner = min(values, key=lambda x: x[1])
        loser = max(values, key=lambda x: x[1])
        improvement = ((loser[1] - winner[1]) / loser[1]) * 100
        print(f"   {metric}: {winner[0]} thắng ({winner[1]} vs {loser[1]}, -{improvement:.1f}%)")

    # Overall winner
    xgb_score = sum(1 for m in ["MAE", "RMSE", "MAPE (%)"] if xgb_test[m] < prophet_test[m])
    prophet_score = 3 - xgb_score

    if xgb_score > prophet_score:
        overall = "🤖 XGBoost"
    elif prophet_score > xgb_score:
        overall = "🔮 Prophet"
    else:
        overall = "🤝 Hòa"

    print(f"\n   🏅 Mô hình tổng thể tốt hơn: {overall} ({xgb_score}-{prophet_score})")

    return df_metrics, all_metrics


def generate_report(merged, corr_results, xgb_results, prophet_results, df_metrics):
    """Tạo báo cáo tổng hợp."""
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    report_path = os.path.join(output_dir, "report.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("  DỰ ÁN ANTIGRAVITY: PHÂN TÍCH LIÊN THỊ TRƯỜNG & DỰ BÁO GIÁ VÀNG")
    lines.append("=" * 70)
    lines.append("")

    # 1. Dữ liệu
    lines.append("1. TỔNG QUAN DỮ LIỆU")
    lines.append("-" * 50)
    lines.append(f"   Số bản ghi: {len(merged)}")
    lines.append(f"   Khoảng thời gian: {merged.index[0].date()} → {merged.index[-1].date()}")
    lines.append(f"   Biến: Gold (GC=F), WTI (CL=F), DXY (DX-Y.NYB)")
    lines.append("")

    # 2. Tương quan
    lines.append("2. PHÂN TÍCH TƯƠNG QUAN")
    lines.append("-" * 50)
    if "pearson" in corr_results:
        lines.append("   Ma trận Pearson:")
        lines.append(corr_results["pearson"].round(4).to_string())
    lines.append("")

    if "granger" in corr_results:
        lines.append("   Granger Causality:")
        for name, res in corr_results["granger"].items():
            if "best_pvalue" in res:
                sig = "CÓ" if res["significant"] else "KHÔNG"
                lines.append(f"   {name}: p={res['best_pvalue']:.6f} (lag={res['best_lag']}) → {sig} nhân quả")
    lines.append("")

    # 3. So sánh mô hình
    lines.append("3. SO SÁNH MÔ HÌNH")
    lines.append("-" * 50)
    lines.append(df_metrics.to_string(index=False))
    lines.append("")

    # 4. XGBoost params
    lines.append("4. THAM SỐ TỐI ƯU XGBOOST")
    lines.append("-" * 50)
    for k, v in xgb_results["best_params"].items():
        lines.append(f"   {k}: {v}")
    lines.append("")

    # 5. Feature importance
    lines.append("5. TOP 10 FEATURES QUAN TRỌNG NHẤT (XGBOOST)")
    lines.append("-" * 50)
    top_feats = xgb_results["feature_importance"].head(10)
    for _, row in top_feats.iterrows():
        lines.append(f"   {row['Feature']:30s} {row['Importance']:.4f}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Ghi chú: Tài liệu này phục vụ cho mục đích học thuật và nghiên cứu")
    lines.append("=" * 70)

    report = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📝 Báo cáo đã được lưu: {report_path}")

    return report_path, report
