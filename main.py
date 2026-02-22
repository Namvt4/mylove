"""
Dự án Antigravity: Phân tích Liên thị trường & Dự báo Giá Vàng
================================================================

Pipeline chính chạy tuần tự cho HAI giai đoạn:
  - Giai đoạn 1: 2014-2019 (Pre-COVID)
  - Giai đoạn 2: 2020-2025 (Post-COVID)

Sau đó so sánh liên giai đoạn.

Tác giả: Dự án Antigravity
Ghi chú: Phục vụ mục đích học thuật và nghiên cứu
"""

import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_collection
import correlation_analysis
import model_xgboost
import model_prophet
import evaluation
import visualizations
import period_comparison

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "output", "figures")


def run_period(period_name, start, end):
    """Chạy toàn bộ pipeline cho một giai đoạn."""
    print("\n" + "#" * 60)
    print(f"#  GIAI ĐOẠN: {period_name} ({start} → {end})")
    print("#" * 60)

    # 1. Data
    merged, df_prophet = data_collection.run(start=start, end=end)

    # 2. Correlation
    corr_results = correlation_analysis.run(merged, prefix=period_name)

    # 3. XGBoost
    xgb_results = model_xgboost.run(merged)

    # 4. Prophet
    prophet_results = model_prophet.run(df_prophet)

    # 5. Evaluation
    df_metrics, all_metrics = evaluation.compare_models(xgb_results, prophet_results)

    # 6. Visualizations
    visualizations.run(merged, xgb_results, prophet_results, df_metrics, prefix=period_name)

    return {
        "merged": merged,
        "df_prophet": df_prophet,
        "corr_results": corr_results,
        "xgb_results": xgb_results,
        "prophet_results": prophet_results,
        "df_metrics": df_metrics,
        "all_metrics": all_metrics,
    }


def main():
    start_time = time.time()

    print("+" + "=" * 58 + "+")
    print("|  DU AN ANTIGRAVITY                                       |")
    print("|  Phan tich Lien thi truong & Du bao Gia Vang             |")
    print("|  Gold x WTI x DXY | XGBoost vs Prophet                  |")
    print("|  So sanh 2 giai doan: 2014-2019 vs 2020-2025             |")
    print("+" + "=" * 58 + "+")

    # ============================================================
    # GIAI ĐOẠN 1: 2014-2019 (Pre-COVID)
    # ============================================================
    results_old = run_period("2014_2019", "2014-01-01", "2019-12-31")

    # ============================================================
    # GIAI ĐOẠN 2: 2020-2025 (Post-COVID)
    # ============================================================
    results_new = run_period("2020_2025", "2020-01-01", "2025-12-31")

    # ============================================================
    # SO SÁNH LIÊN GIAI ĐOẠN
    # ============================================================
    comparison = period_comparison.run(
        corr_old=results_old["corr_results"],
        corr_new=results_new["corr_results"],
        metrics_old=results_old["all_metrics"],
        metrics_new=results_new["all_metrics"],
        merged_old=results_old["merged"],
        merged_new=results_new["merged"],
    )

    # ============================================================
    # BÁO CÁO TỔNG HỢP
    # ============================================================
    evaluation.generate_report_multi(
        results_old=results_old,
        results_new=results_new,
        comparison=comparison,
    )

    # ============================================================
    # TỔNG KẾT
    # ============================================================
    elapsed = time.time() - start_time
    print("\n+" + "=" * 58 + "+")
    print("|  HOAN THANH!                                             |")
    print("+" + "=" * 58 + "+")
    print(f"\n  Thoi gian chay: {elapsed:.1f} giay")
    print(f"  Ket qua trong: {os.path.join(os.path.dirname(__file__), 'output')}")
    print(f"  Bieu do:  output/figures/")
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
