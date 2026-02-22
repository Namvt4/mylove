"""
Giai đoạn 2: Phân tích Tương quan Nâng cao
- Static Correlation (Pearson)
- Rolling Correlation (30-ngày, 60-ngày)
- Granger Causality Test
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import os
import warnings

warnings.filterwarnings("ignore")

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "output", "figures")


def static_correlation(merged, prefix=""):
    """Tính ma trận tương quan Pearson."""
    corr_matrix = merged.corr(method="pearson")

    print("\n📈 Ma trận tương quan Pearson:")
    print(corr_matrix.round(4).to_string())

    # Vẽ heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".4f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    title_suffix = f" [{prefix}]" if prefix else ""
    ax.set_title(f"Ma trận Tương quan Pearson\n(Gold - WTI - DXY){title_suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fname = f"{prefix}_pearson_correlation_heatmap.png" if prefix else "pearson_correlation_heatmap.png"
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 Đã lưu: {path}")

    return corr_matrix


def rolling_correlation(merged, windows=[30, 60], prefix=""):
    """Tính rolling correlation giữa Gold-DXY và Gold-WTI."""
    fig, axes = plt.subplots(len(windows), 1, figsize=(14, 5 * len(windows)), sharex=True)
    if len(windows) == 1:
        axes = [axes]

    results = {}

    for idx, window in enumerate(windows):
        ax = axes[idx]

        # Gold vs DXY
        roll_dxy = merged["Gold"].rolling(window).corr(merged["DXY"])
        # Gold vs WTI
        roll_wti = merged["Gold"].rolling(window).corr(merged["WTI"])

        results[f"Gold_DXY_{window}d"] = roll_dxy
        results[f"Gold_WTI_{window}d"] = roll_wti

        ax.plot(roll_dxy.index, roll_dxy, label=f"Gold ↔ DXY", color="#e74c3c", linewidth=1.5)
        ax.plot(roll_wti.index, roll_wti, label=f"Gold ↔ WTI", color="#2ecc71", linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(roll_dxy.index, roll_dxy, alpha=0.1, color="#e74c3c")
        ax.fill_between(roll_wti.index, roll_wti, alpha=0.1, color="#2ecc71")
        ax.set_ylabel("Hệ số tương quan", fontsize=11)
        ax.set_title(f"Rolling Correlation - Cửa sổ {window} ngày", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Thời gian", fontsize=11)
    plt.tight_layout()

    fname = f"{prefix}_rolling_correlation.png" if prefix else "rolling_correlation.png"
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📉 Rolling Correlation đã được tính và lưu: {path}")

    return results


def granger_causality(merged, maxlag=10, prefix=""):
    """Kiểm tra Granger Causality: DXY/WTI → Gold."""
    print("\n🔍 Kiểm tra Granger Causality:")
    print("-" * 50)

    results = {}

    pairs = [
        ("DXY → Gold", ["Gold", "DXY"]),
        ("WTI → Gold", ["Gold", "WTI"]),
    ]

    for name, cols in pairs:
        print(f"\n  📌 {name} (maxlag={maxlag}):")
        data = merged[cols].dropna()

        try:
            gc_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)

            # Tìm lag tốt nhất (p-value nhỏ nhất)
            best_lag = None
            best_pvalue = 1.0

            lag_results = []
            for lag in range(1, maxlag + 1):
                p_val = gc_result[lag][0]["ssr_ftest"][1]
                lag_results.append({"lag": lag, "p_value": p_val})
                if p_val < best_pvalue:
                    best_pvalue = p_val
                    best_lag = lag

            results[name] = {
                "best_lag": best_lag,
                "best_pvalue": best_pvalue,
                "significant": best_pvalue < 0.05,
                "details": lag_results,
            }

            status = "✅ CÓ" if best_pvalue < 0.05 else "❌ KHÔNG"
            print(f"     Lag tốt nhất: {best_lag} | p-value: {best_pvalue:.6f}")
            print(f"     Kết luận: {status} có quan hệ nhân quả Granger (α=0.05)")

        except Exception as e:
            print(f"     ⚠️ Lỗi: {e}")
            results[name] = {"error": str(e)}

    # Vẽ biểu đồ p-value theo lag
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, cols) in enumerate(pairs):
        ax = axes[idx]
        if name in results and "details" in results[name]:
            details = results[name]["details"]
            lags = [d["lag"] for d in details]
            pvals = [d["p_value"] for d in details]

            bars = ax.bar(lags, pvals, color=["#2ecc71" if p < 0.05 else "#e74c3c" for p in pvals], alpha=0.8)
            ax.axhline(y=0.05, color="red", linestyle="--", linewidth=1.5, label="α = 0.05")
            ax.set_xlabel("Lag (ngày)", fontsize=11)
            ax.set_ylabel("p-value", fontsize=11)
            ax.set_title(f"Granger Causality: {name}", fontsize=13, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = f"{prefix}_granger_causality.png" if prefix else "granger_causality.png"
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   💾 Biểu đồ Granger Causality: {path}")

    return results


def run(merged, prefix=""):
    """Pipeline phân tích tương quan."""
    print("\n" + "=" * 60)
    print("📊 GIAI ĐOẠN 2: PHÂN TÍCH TƯƠNG QUAN NÂNG CAO")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    corr_matrix = static_correlation(merged, prefix=prefix)
    rolling_results = rolling_correlation(merged, prefix=prefix)
    granger_results = granger_causality(merged, prefix=prefix)

    return {
        "pearson": corr_matrix,
        "rolling": rolling_results,
        "granger": granger_results,
    }


if __name__ == "__main__":
    import data_collection
    merged, _ = data_collection.run()
    run(merged)
