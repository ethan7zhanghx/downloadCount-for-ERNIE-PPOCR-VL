#!/usr/bin/env python3
"""分析1.16数据差异问题"""
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ernie_tracker.config import DB_PATH, DATA_TABLE
from ernie_tracker.db import load_data_from_db
from ernie_tracker.analysis import calculate_weekly_report
import pandas as pd

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 查看回填记录的数量和日期分布
    print("="*60)
    print("1. 查看回填记录（download_count='0' 且 date < '2026-01-16'）")
    print("="*60)
    cursor.execute(f"""
        SELECT date, repo, COUNT(*) as count
        FROM {DATA_TABLE}
        WHERE download_count = '0'
        AND date < '2026-01-16'
        AND repo IN ('AI Studio', 'ModelScope')
        AND model_category IN ('ernie-4.5', 'paddleocr-vl')
        GROUP BY date, repo
        ORDER BY date, repo
    """)
    backfill_records = cursor.fetchall()
    print(f"{'日期':<15} {'平台':<15} {'数量':>10}")
    print("-"*60)
    for date, repo, count in backfill_records:
        print(f"{date:<15} {repo:<15} {count:>10}")

    total_backfill = sum(r[2] for r in backfill_records)
    print(f"\n总计回填记录数: {total_backfill}")

    # 2. 查看1.16当天的AI Studio数据（包含简化格式）
    print("\n" + "="*60)
    print("2. 1.16当天AI Studio数据（可能有简化格式）")
    print("="*60)
    cursor.execute(f"""
        SELECT model_name, download_count
        FROM {DATA_TABLE}
        WHERE date = '2026-01-16'
        AND repo = 'AI Studio'
        ORDER BY rowid DESC
        LIMIT 50
    """)
    ai_studio_records = cursor.fetchall()
    print(f"{'模型名称':<40} {'下载量':<15}")
    print("-"*60)
    has_simplified = False
    for model_name, count in ai_studio_records:
        is_simplified = any(count.endswith(suffix) for suffix in ['k', 'K', 'w', 'W', '万', 'M', 'm'])
        if is_simplified:
            has_simplified = True
            print(f"{model_name:<40} {count:<15} ⚠️ 简化格式")
        elif len(ai_studio_records) < 20:  # 只显示前20个
            print(f"{model_name:<40} {count:<15}")

    if has_simplified:
        print("\n⚠️ 发现简化格式数据！")
    else:
        print("\n✅ 未发现简化格式数据")

    # 3. 检查AI Studio中哪些模型有简化格式
    print("\n" + "="*60)
    print("3. 检查1.16当天所有AI Studio模型的下载量格式")
    print("="*60)
    cursor.execute(f"""
        SELECT download_count, COUNT(*) as count
        FROM {DATA_TABLE}
        WHERE date = '2026-01-16'
        AND repo = 'AI Studio'
        GROUP BY download_count
        ORDER BY count DESC
    """)
    format_stats = cursor.fetchall()
    print(f"{'下载量':<20} {'数量':>10}")
    print("-"*40)
    for value, count in format_stats:
        # 检查是否是简化格式
        is_simplified = any(str(value).endswith(suffix) for suffix in ['k', 'K', 'w', 'W', '万', 'M', 'm'])
        marker = " ⚠️ 简化" if is_simplified else ""
        print(f"{str(value):<20} {count:>10}{marker}")

    # 4. 使用实际代码计算1.16数据（上周的值）
    print("\n" + "="*60)
    print("4. 使用 calculate_weekly_report 计算1.16数据")
    print("="*60)
    try:
        result = calculate_weekly_report(
            current_date='2026-01-16',
            previous_date='2026-01-09',
            model_order=['ernie-4.5']
        )

        if result and 'total_summary' in result:
            summary = result['total_summary']
            print(f"官方模型总量: {summary.get('official_current_total', 0):,}")
            print(f"衍生模型总量: {summary.get('derivative_current_total', 0):,}")
            print(f"总计: {summary.get('all_current_total', 0):,} ({summary.get('all_current_total', 0) / 10000:.2f}万)")
    except Exception as e:
        print(f"计算失败: {e}")
        import traceback
        traceback.print_exc()

    # 5. 检查是否有负增长重新获取的记录
    print("\n" + "="*60)
    print("5. 检查是否有负增长重新获取的记录（同一天多条记录）")
    print("="*60)
    cursor.execute(f"""
        SELECT date, repo, model_name, COUNT(*) as count
        FROM {DATA_TABLE}
        WHERE date = '2026-01-16'
        AND repo = 'AI Studio'
        GROUP BY date, repo, model_name
        HAVING count > 1
        ORDER BY count DESC
        LIMIT 10
    """)
    duplicate_records = cursor.fetchall()
    if duplicate_records:
        print(f"{'日期':<15} {'平台':<15} {'模型':<40} {'记录数':>10}")
        print("-"*80)
        for date, repo, model_name, count in duplicate_records:
            print(f"{date:<15} {repo:<15} {model_name:<40} {count:>10}")
    else:
        print("✅ 未发现同一天多条记录")

    conn.close()

if __name__ == '__main__':
    main()
