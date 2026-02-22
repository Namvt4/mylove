"""
Dự án Antigravity: Phân tích Liên thị trường & Dự báo Giá Vàng
================================================================

Pipeline chính chạy tuần tự:
1. Thu thập dữ liệu (Yahoo Finance)
2. Phân tích tương quan (Pearson, Rolling, Granger)
3. XGBoost training (+ Optuna optimization)
4. Prophet training (+ Regressors)
5. Evaluation & Comparison
6. Visualization & Report

Tác giả: Dự án Antigravity
Ghi chú: Phục vụ mục đích học thuật và nghiên cứu
"""

import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")

# Thêm thư mục dự án vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_collection
import correlation_analysis
import model_xgboost
import model_prophet
import evaluation
import visualizations


def main():
    start_time = time.time()

    print("╔" + "═" * 58 + "╗")
    print("║  DỰ ÁN ANTIGRAVITY                                      ║")
    print("║  Phân tích Liên thị trường & Dự báo Giá Vàng            ║")
    print("║  Gold × WTI × DXY | XGBoost vs Prophet                  ║")
    print("╚" + "═" * 58 + "╝")

    # ============================================================
    # GIAI ĐOẠN 1: Thu thập & Tiền xử lý dữ liệu
    # ============================================================
    merged, df_prophet = data_collection.run()

    # ============================================================
    # GIAI ĐOẠN 2: Phân tích tương quan
    # ============================================================
    corr_results = correlation_analysis.run(merged)

    # ============================================================
    # GIAI ĐOẠN 3A: XGBoost + Optuna
    # ============================================================
    xgb_results = model_xgboost.run(merged)

    # ============================================================
    # GIAI ĐOẠN 3B: Prophet
    # ============================================================
    prophet_results = model_prophet.run(df_prophet)

    # ============================================================
    # GIAI ĐOẠN 4: Đánh giá & So sánh
    # ============================================================
    df_metrics, all_metrics = evaluation.compare_models(xgb_results, prophet_results)

    # ============================================================
    # GIAI ĐOẠN 5: Visualizations
    # ============================================================
    visualizations.run(merged, xgb_results, prophet_results, df_metrics)

    # ============================================================
    # GIAI ĐOẠN 6: Báo cáo
    # ============================================================
    report_path, report = evaluation.generate_report(
        merged, corr_results, xgb_results, prophet_results, df_metrics
    )

    # ============================================================
    # TỔNG KẾT
    # ============================================================
    elapsed = time.time() - start_time
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  ✅ HOÀN THÀNH!                                          ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n⏱️  Thời gian chạy: {elapsed:.1f} giây")
    print(f"📁 Kết quả trong: {os.path.join(os.path.dirname(__file__), 'output')}")
    print(f"📊 Biểu đồ:  output/figures/")
    print(f"📝 Báo cáo:  {report_path}")
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
