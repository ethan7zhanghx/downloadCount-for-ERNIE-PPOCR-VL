#!/usr/bin/env python3
"""对比上周和本周的1.16数据计算结果"""
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ernie_tracker.config import DB_PATH, DATA_TABLE
from ernie_tracker.db import load_data_from_db
from ernie_tracker.analysis import filter_by_series, mark_official_models
import pandas as pd

def calculate_like_old_code(current_date, previous_date):
    """使用旧逻辑计算"""
    full_data = load_data_from_db(date_filter=None, last_value_per_model=False)
    full_data = filter_by_series(full_data)

    if not full_data.empty:
        mark_official_models(full_data)
        current_dt = pd.to_datetime(current_date)
        previous_dt = pd.to_datetime(previous_date)

        def platform_peak_total(df, cutoff_dt, is_official=None):
            """按平台统计历史峰值（可分别统计官方或衍生模型）"""
            if is_official is not None:
                subset = df[(df['is_official'] == is_official) & (df['date'] <= cutoff_dt)]
            else:
                subset = df[df['date'] <= cutoff_dt]
            if subset.empty:
                return pd.Series()
            peak_per_model = subset.groupby(['repo', 'publisher', 'model_name'])['download_count'].max()
            platform_totals = peak_per_model.groupby('repo').sum()
            return platform_totals

        # 分别统计当前和官方+衍生模型的历史峰值
        current_official_platforms = platform_peak_total(full_data, current_dt, is_official=True)
        current_derivative_platforms = platform_peak_total(full_data, current_dt, is_official=False)
        current_platform_totals = current_official_platforms.add(current_derivative_platforms, fill_value=0)

        # 计算总体
        official_total = current_official_platforms.sum()
        derivative_total = current_derivative_platforms.sum()
        all_total = current_platform_totals.sum()

        return {
            'official': official_total,
            'derivative': derivative_total,
            'all': all_total,
            'by_platform': current_platform_totals.to_dict()
        }
    else:
        return {'official': 0, 'derivative': 0, 'all': 0, 'by_platform': {}}

def calculate_like_new_code(current_date, previous_date):
    """使用新逻辑计算"""
    full_data = load_data_from_db(date_filter=None, last_value_per_model=False)
    full_data = filter_by_series(full_data)

    if not full_data.empty:
        mark_official_models(full_data)
        current_dt = pd.to_datetime(current_date)
        previous_dt = pd.to_datetime(previous_date)

        def peak_total_by_type(df, cutoff_dt, is_official):
            """统计官方或衍生模型的历史峰值总和（统一逻辑）"""
            subset = df[(df['is_official'] == is_official) & (df['date'] <= cutoff_dt)]
            if subset.empty:
                return 0
            peak_per_combo = subset.groupby(['repo', 'publisher', 'model_name'])['download_count'].max()
            return peak_per_combo.sum()

        def peak_total_by_platform(df, cutoff_dt, is_official):
            """统计官方或衍生模型按平台的历史峰值"""
            subset = df[(df['is_official'] == is_official) & (df['date'] <= cutoff_dt)]
            if subset.empty:
                return pd.Series()
            peak_per_model = subset.groupby(['repo', 'publisher', 'model_name'])['download_count'].max()
            platform_totals = peak_per_model.groupby('repo').sum()
            return platform_totals

        # 分别统计官方和衍生模型
        official_current_total = peak_total_by_type(full_data, current_dt, is_official=True)
        derivative_current_total = peak_total_by_type(full_data, current_dt, is_official=False)
        all_current_total = official_current_total + derivative_current_total

        # 按平台统计
        current_official_platforms = peak_total_by_platform(full_data, current_dt, is_official=True)
        current_derivative_platforms = peak_total_by_platform(full_data, current_dt, is_official=False)
        current_platform_totals = current_official_platforms.add(current_derivative_platforms, fill_value=0)

        return {
            'official': official_current_total,
            'derivative': derivative_current_total,
            'all': all_current_total,
            'by_platform': current_platform_totals.to_dict()
        }
    else:
        return {'official': 0, 'derivative': 0, 'all': 0, 'by_platform': {}}

def main():
    print("="*80)
    print("对比旧逻辑和新逻辑的1.16数据计算结果")
    print("="*80)

    old_result = calculate_like_old_code('2026-01-16', '2026-01-09')
    new_result = calculate_like_new_code('2026-01-16', '2026-01-09')

    print(f"\n{'指标':<20} {'旧逻辑':>20} {'新逻辑':>20} {'差异':>20}")
    print("-"*80)
    print(f"{'官方模型':<20} {old_result['official']:>20,} {new_result['official']:>20,} {new_result['official'] - old_result['official']:>20,}")
    print(f"{'衍生模型':<20} {old_result['derivative']:>20,} {new_result['derivative']:>20,} {new_result['derivative'] - old_result['derivative']:>20,}")
    print(f"{'总计':<20} {old_result['all']:>20,} {new_result['all']:>20,} {new_result['all'] - old_result['all']:>20,}")

    print(f"\n总计（万）:")
    print(f"  旧逻辑: {old_result['all'] / 10000:.2f}万")
    print(f"  新逻辑: {new_result['all'] / 10000:.2f}万")
    print(f"  差异: {(new_result['all'] - old_result['all']) / 10000:.2f}万")

    # 按平台对比
    print(f"\n{'='*80}")
    print("按平台对比")
    print(f"{'='*80}")
    print(f"{'平台':<20} {'旧逻辑':>20} {'新逻辑':>20} {'差异':>20}")
    print("-"*80)

    all_platforms = set(old_result['by_platform'].keys()) | set(new_result['by_platform'].keys())
    for platform in sorted(all_platforms):
        old_val = old_result['by_platform'].get(platform, 0)
        new_val = new_result['by_platform'].get(platform, 0)
        diff = new_val - old_val
        print(f"{platform:<20} {old_val:>20,} {new_val:>20,} {diff:>20,}")

if __name__ == '__main__':
    main()
