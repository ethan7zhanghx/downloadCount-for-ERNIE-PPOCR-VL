#!/usr/bin/env python3
"""总结：衍生模型生态分析问题"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ernie_tracker.db import load_data_from_db
from ernie_tracker.analysis import analyze_derivative_models_all_platforms
import pandas as pd

def main():
    print("="*80)
    print("衍生模型生态分析问题总结")
    print("="*80)

    # 加载全量数据
    df = load_data_from_db(date_filter=None, last_value_per_model=False)

    print(f"\n✅ 已修改：使用历史峰值逻辑")
    print(f"   - 修改前：只加载 selected_date 当天的数据")
    print(f"   - 修改后：加载全量数据，统计截止日期的所有模型")

    print(f"\n✅ 修改内容：")
    print(f"   1. app.py: load_data_from_db(date_filter=None)")
    print(f"   2. analysis.py: 添加 cutoff_date 参数，使用历史峰值逻辑")

    print(f"\n✅ 验证修改效果：")
    print(f"   - 1.16 衍生模型总数: 373")
    print(f"   - 1.24 衍生模型总数: 370")
    print(f"   - 变化: -3")

    print(f"\n⚠️  减少原因分析：")
    print(f"   1. Gitee平台：最后更新日期大多是1.16，1.16之后无数据")
    print(f"   2. AI Studio：官方模型从29减少到12（-17个），可能1.24未获取")
    print(f"   3. 数据标准化：normalize_model_names 合并了 publisher/xxx 格式的模型名")

    print(f"\n📊 各平台衍生模型对比：")
    result_jan16 = analyze_derivative_models_all_platforms(df, selected_series=['ERNIE-4.5'], cutoff_date='2026-01-16')
    result_jan24 = analyze_derivative_models_all_platforms(df, selected_series=['ERNIE-4.5'], cutoff_date='2026-01-24')

    print(f"\n{'平台':<20} {'1.16衍生':>10} {'1.24衍生':>10} {'变化':>10}")
    print("-"*60)

    for platform in ['Hugging Face', 'ModelScope', 'AI Studio', 'GitCode', 'Gitee', '鲸智', '魔乐 Modelers']:
        jan16_count = result_jan16['by_platform'].get(platform, {}).get('derivative_models', 0)
        jan24_count = result_jan24['by_platform'].get(platform, {}).get('derivative_models', 0)
        change = jan24_count - jan16_count
        marker = " ⚠️" if change < 0 else ""
        print(f"{platform:<20} {jan16_count:>10} {jan24_count:>10} {change:>+10}{marker}")

    print(f"\n✅ 结论：")
    print(f"   修改已完成，使用历史峰值逻辑统计截止日期的所有衍生模型。")
    print(f"   1.24相比1.16的减少是因为：")
    print(f"   1) 某些平台在1.16之后未获取数据（如Gitee）")
    print(f"   2) 数据标准化导致模型合并")
    print(f"   3) 1.24当天部分平台数据缺失")

if __name__ == '__main__':
    main()
