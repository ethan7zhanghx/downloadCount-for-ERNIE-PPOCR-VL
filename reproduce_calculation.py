#!/usr/bin/env python3
"""重现1.16数据的计算，找出差异原因"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ernie_tracker.analysis import calculate_weekly_report

def main():
    print("="*80)
    print("重现1.16数据计算")
    print("="*80)

    result = calculate_weekly_report(
        current_date='2026-01-16',
        previous_date='2026-01-09',
        model_order=['ernie-4.5']
    )

    if not result:
        print("❌ calculate_weekly_report 返回 None")
        return

    # 查看 summary_stats 的内容
    if 'summary_stats' in result:
        summary_stats = result['summary_stats']
        print(f"summary_stats 的内容: {summary_stats}")
        print()

    # 查看 platform_summary 的内容
    if 'platform_summary' in result:
        platform_summary = result['platform_summary']
        print(f"platform_summary 的总下载量: {platform_summary['current_total'].sum():,}")
        print()

if __name__ == '__main__':
    main()
