#!/usr/bin/env python3
"""检查1.16当天官方模型的数据情况"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ernie_tracker.db import load_data_from_db
from ernie_tracker.analysis import mark_official_models
import pandas as pd

def main():
    print("="*80)
    print("检查1.16当天官方模型数据")
    print("="*80)

    # 加载数据
    data = load_data_from_db(date_filter=None, last_value_per_model=False)
    data['download_count'] = pd.to_numeric(data['download_count'], errors='coerce').fillna(0)

    # 筛选ERNIE-4.5系列
    ernie_data = data[
        (data['model_category'] == 'ernie-4.5') |
        (data['model_name'].str.contains('ERNIE-4.5', case=False, na=False))
    ].copy()

    # 标记官方模型（注意：mark_official_models返回新的DataFrame）
    ernie_data = mark_official_models(ernie_data)

    # 筛选官方模型
    official_data = ernie_data[ernie_data['is_official'] == True].copy()

    # 1.16当天的官方模型
    official_jan16 = official_data[official_data['date'] == '2026-01-16']
    jan16_total = official_jan16['download_count'].sum()

    print(f"\n📊 1.16当天官方模型（当日值）:")
    print(f"  总下载量: {jan16_total:,} ({jan16_total / 10000:.2f}万)")
    print(f"  模型数量: {len(official_jan16)}")

    # 历史最大值
    official_before_jan16 = official_data[official_data['date'] <= '2026-01-16']
    peak_per_model = official_before_jan16.groupby(['repo', 'publisher', 'model_name'])['download_count'].max()
    peak_total = peak_per_model.sum()

    print(f"\n📊 官方模型历史最大值（截止1.16）:")
    print(f"  总下载量: {peak_total:,} ({peak_total / 10000:.2f}万)")
    print(f"  模型数量: {len(peak_per_model)}")

    # 差异
    diff = peak_total - jan16_total
    print(f"\n📈 差异分析:")
    print(f"  当日值: {jan16_total:,} ({jan16_total / 10000:.2f}万)")
    print(f"  历史峰值: {peak_total:,} ({peak_total / 10000:.2f}万)")
    print(f"  差异: {diff:+,} ({diff / 10000:+.2f}万)")

    # 找出1.16当天缺失的官方模型
    all_official_models = set(official_before_jan16.groupby(['repo', 'publisher', 'model_name']).groups.keys())
    jan16_models = set(official_jan16.groupby(['repo', 'publisher', 'model_name']).groups.keys())
    missing_models = all_official_models - jan16_models

    if missing_models:
        print(f"\n⚠️ 1.16当天缺失的官方模型 ({len(missing_models)}个):")
        for repo, publisher, model_name in sorted(missing_models):
            print(f"  {repo} - {publisher}/{model_name}")
    else:
        print(f"\n✅ 1.16当天没有缺失的官方模型")

    # 找出历史峰值大于当日值的模型
    print(f"\n📊 历史峰值 > 当日值的模型:")
    model_comparison = []
    for (repo, publisher, model_name), group in official_before_jan16.groupby(['repo', 'publisher', 'model_name']):
        peak = group['download_count'].max()
        jan16_value = group[group['date'] == '2026-01-16']['download_count'].sum()
        if jan16_value == 0:
            jan16_value = None  # 当天没有数据
        if jan16_value is not None and peak > jan16_value:
            model_comparison.append({
                'repo': repo,
                'publisher': publisher,
                'model_name': model_name,
                'jan16_value': jan16_value,
                'peak': peak,
                'diff': peak - jan16_value
            })

    # 按差异排序
    model_comparison.sort(key=lambda x: x['diff'], reverse=True)

    if model_comparison:
        print(f"{'平台':<15} {'模型':<50} {'当日值':>12} {'历史峰值':>12} {'差异':>12}")
        print("-"*90)
        total_diff = 0
        for item in model_comparison[:20]:  # 只显示前20个
            print(f"{item['repo']:<15} {item['model_name']:<50} {item['jan16_value']:>12,} {item['peak']:>12,} {item['diff']:>12,}")
            total_diff += item['diff']
        if len(model_comparison) > 20:
            print(f"... 还有 {len(model_comparison) - 20} 个模型")
        print(f"\n总差异: {total_diff:,} ({total_diff / 10000:.2f}万)")
    else:
        print("✅ 没有历史峰值大于当日值的模型")

if __name__ == '__main__':
    main()
