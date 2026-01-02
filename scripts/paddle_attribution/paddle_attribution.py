#!/usr/bin/env python
"""
Paddle attribution helper:
- 输出 Q4 每周（周五）Paddle 尾缀模型的累计与增量
- 绘制增长曲线
- 基于已标注的 PT/Paddle 模型下载量比例，估算未标注 safetensors 模型的 Paddle 占比
"""

from __future__ import annotations

import ast
import os
import sqlite3
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / "mpl_config")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rcParams

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "ernie_downloads.db"
OUT_DIR = Path(__file__).parent
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# 优先尝试中文字体，避免中文标签缺字
rcParams["font.sans-serif"] = [
    "PingFang HK",
    "PingFang SC",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "SimHei",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


def load_downloads() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT date, repo, model_name, download_count, tags FROM model_downloads",
            conn,
        )
    df["date"] = pd.to_datetime(df["date"])
    df["download_count"] = (
        pd.to_numeric(df["download_count"], errors="coerce").fillna(0).astype(float)
    )
    return df


def dedup_daily_max(df: pd.DataFrame) -> pd.DataFrame:
    """同一天同平台同模型取最大快照，避免重复行影响汇总。"""
    cols = ["date", "repo", "model_name"]
    deduped = (
        df.sort_values(cols + ["download_count"])
        .groupby(cols, as_index=False)
        .agg(
            download_count=("download_count", "max"),
            tags=("tags", "last"),
        )
    )
    return deduped


def build_friday_range(start: str, end: str) -> List[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="W-FRI"))


def compute_weekly_cumulative(
    df: pd.DataFrame, fridays: Iterable[pd.Timestamp]
) -> pd.DataFrame:
    """对每个 repo+模型做前向填充后汇总为周五累计。"""
    fridays = pd.to_datetime(list(fridays))
    records = []
    for (repo, model), grp in df.groupby(["repo", "model_name"]):
        series = (
            grp.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")["download_count"]
        )
        series = series.cummax()
        filled = series.reindex(fridays, method="ffill").fillna(0)
        records.append(filled.to_frame(name=f"{repo}|{model}"))

    combined = pd.concat(records, axis=1)
    summed = combined.sum(axis=1)
    weekly = pd.DataFrame({"friday": summed.index, "cumulative": summed.values})
    weekly["weekly_increment"] = weekly["cumulative"].diff()
    return weekly


def plot_growth(df: pd.DataFrame, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["friday"], df["cumulative"], marker="o", linewidth=2, color="#3b7ddd")
    ax.set_title("Paddle权重模型累计下载量")
    ax.set_xlabel("")
    ax.set_ylabel("累计下载量")
    ymin = df["cumulative"].min()
    ymax = df["cumulative"].max()
    margin = max((ymax - ymin) * 0.08, 500)  # 适当放大 y 轴便于观察增长
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def has_safetensors(tag_value: str | None) -> bool:
    if not tag_value:
        return False
    try:
        parsed = ast.literal_eval(tag_value)
        if isinstance(parsed, list):
            return any(isinstance(t, str) and "safetensor" in t.lower() for t in parsed)
        if isinstance(parsed, str):
            return "safetensor" in parsed.lower()
    except Exception:
        if isinstance(tag_value, str) and "safetensor" in tag_value.lower():
            return True
    return False


OFFICIAL_MODELS = {
    "ERNIE-4.5-0.3B-PT",
    "ERNIE-4.5-0.3B-Paddle",
    "ERNIE-4.5-0.3B-Base-PT",
    "ERNIE-4.5-0.3B-Base-Paddle",
    "ERNIE-4.5-21B-A3B-PT",
    "ERNIE-4.5-21B-A3B-Paddle",
    "ERNIE-4.5-21B-A3B-Base-PT",
    "ERNIE-4.5-21B-A3B-Base-Paddle",
    "ERNIE-4.5-21B-A3B-Thinking",
    "ERNIE-4.5-300B-A47B-PT",
    "ERNIE-4.5-300B-A47B-Paddle",
    "ERNIE-4.5-300B-A47B-Base-PT",
    "ERNIE-4.5-300B-A47B-Base-Paddle",
    "ERNIE-4.5-300B-A47B-FP8-Paddle",
    "ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle",
    "ERNIE-4.5-300B-A47B-2Bits-Paddle",
    "ERNIE-4.5-300B-A47B-2Bits-TP2-Paddle",
    "ERNIE-4.5-300B-A47B-2Bits-TP4-Paddle",
    "ERNIE-4.5-VL-28B-A3B-PT",
    "ERNIE-4.5-VL-28B-A3B-Paddle",
    "ERNIE-4.5-VL-28B-A3B-Base-PT",
    "ERNIE-4.5-VL-28B-A3B-Base-Paddle",
    "ERNIE-4.5-VL-28B-A3B-Thinking",
    "ERNIE-4.5-VL-424B-A47B-PT",
    "ERNIE-4.5-VL-424B-A47B-Paddle",
    "ERNIE-4.5-VL-424B-A47B-Base-PT",
    "ERNIE-4.5-VL-424B-A47B-Base-Paddle",
    "PaddleOCR-VL",
}


def compute_q4_increment(series: pd.Series) -> float:
    """Q4 新增：末值 - 首个非零（若全 0，则为 0）。"""
    nonzero = series[series > 0]
    start_val = nonzero.iloc[0] if not nonzero.empty else 0
    end_val = series.iloc[-1] if not series.empty else 0
    return end_val - start_val


def estimate_paddle_share(df: pd.DataFrame, fridays: Iterable[pd.Timestamp]) -> None:
    """基于“总量占比”，去估算“Q4 未标注 safetensors 的新增中，Paddle 贡献”."""
    fridays = pd.to_datetime(list(fridays))
    official = df[df["model_name"].isin(OFFICIAL_MODELS)].copy()
    official["is_paddle"] = official["model_name"].str.endswith("Paddle")
    official["is_pt"] = official["model_name"].str.endswith("PT")

    def sum_latest(sub: pd.DataFrame) -> float:
        """每个 repo+模型取最大快照后求和，代表当前总量。"""
        if sub.empty:
            return 0.0
        latest = (
            sub.sort_values(["date", "download_count"])
            .groupby(["repo", "model_name"], as_index=False)
            .agg(download_count=("download_count", "max"))
        )
        return latest["download_count"].sum()

    def weekly_cum(sub: pd.DataFrame) -> pd.Series:
        """按周五对某子集做前向填充+cummax，再汇总。"""
        records = []
        for (repo, model), grp in sub.groupby(["repo", "model_name"]):
            series = (
                grp.sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .set_index("date")["download_count"]
            ).cummax()
            filled = series.reindex(fridays, method="ffill").fillna(0)
            records.append(filled.to_frame(name=f"{repo}|{model}"))
        if not records:
            return pd.Series([0] * len(fridays), index=fridays)
        combined = pd.concat(records, axis=1)
        return combined.sum(axis=1)

    paddle_total = sum_latest(official[official["is_paddle"]])
    pt_total = sum_latest(official[official["is_pt"]])
    paddle_share = paddle_total / (paddle_total + pt_total) if (paddle_total + pt_total) else 0

    # 官方未标注后缀且带 safetensors
    official_unlabeled = official[~(official["is_paddle"] | official["is_pt"])].copy()
    official_unlabeled["is_safetensors"] = official_unlabeled["tags"].apply(
        has_safetensors
    )
    safetensor_official_unlabeled = official_unlabeled[
        official_unlabeled["is_safetensors"]
    ]
    unlabeled_cum = weekly_cum(safetensor_official_unlabeled)
    unlabeled_inc = compute_q4_increment(unlabeled_cum)
    estimated_paddle_from_unlabeled = unlabeled_inc * paddle_share

    summary = pd.DataFrame(
        [
            {"场景": "官方后缀 Paddle 累计", "下载量": paddle_total},
            {"场景": "官方后缀 PT 累计", "下载量": pt_total},
            {"场景": "Paddle 占比(总量)", "下载量": paddle_share},
            {"场景": "未标注官方 safetensors Q4 新增", "下载量": unlabeled_inc},
            {
                "场景": "估算未标注 safetensors 中 Paddle",
                "下载量": estimated_paddle_from_unlabeled,
            },
            {
                "场景": "Q4 估算 Paddle 总新增",
                "下载量": compute_q4_increment(weekly_cum(official[official["is_paddle"]]))
                + estimated_paddle_from_unlabeled,
            },
        ]
    )
    summary.to_csv(OUT_DIR / "paddle_share_estimation.csv", index=False)

    print("\n[基于总量占比估算 Q4 未标注贡献]")
    print(f"Paddle 后缀累计: {paddle_total:,.0f}")
    print(f"PT 后缀累计: {pt_total:,.0f}")
    print(f"Paddle 占比(总量): {paddle_share:.3%}")
    print(
        f"未标注官方 safetensors Q4 新增: {unlabeled_inc:,.0f} -> "
        f"估算其中 Paddle: {estimated_paddle_from_unlabeled:,.0f}"
    )


def main() -> None:
    df = load_downloads()
    daily = dedup_daily_max(df)

    q4_mask = (daily["date"] >= "2025-10-11") & (daily["date"] <= "2025-12-31")
    paddle_daily_q4 = daily[q4_mask & daily["model_name"].str.endswith("Paddle")]

    if paddle_daily_q4.empty:
        raise SystemExit("No Paddle data found for Q4.")

    first_date = paddle_daily_q4["date"].min()
    start_friday = (
        first_date
        if first_date.weekday() == 4
        else first_date + pd.offsets.Week(weekday=4)
    )

    fridays = build_friday_range(start_friday, "2025-12-31")
    weekly = compute_weekly_cumulative(paddle_daily_q4, fridays)
    weekly.to_csv(OUT_DIR / "paddle_q4_weekly.csv", index=False)
    plot_growth(weekly, OUT_DIR / "paddle_q4_weekly.png")

    print("Paddle 尾缀 Q4 周五累计与周增：")
    print(weekly.to_string(index=False, formatters={"cumulative": "{:,.0f}".format}))

    estimate_paddle_share(daily, fridays)


if __name__ == "__main__":
    main()
