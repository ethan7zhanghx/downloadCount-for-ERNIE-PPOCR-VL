#!/usr/bin/env python3
"""分析1.16的重复记录，找出数据差异原因"""
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ernie_tracker.config import DB_PATH, DATA_TABLE

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 查找1.16当天有多条记录的模型
    print("="*80)
    print("1.16当天有多条记录的AI Studio模型")
    print("="*80)

    cursor.execute(f"""
        SELECT model_name, publisher
        FROM {DATA_TABLE}
        WHERE date = '2026-01-16'
        AND repo = 'AI Studio'
        GROUP BY model_name, publisher
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC
    """)
    dup_models = cursor.fetchall()

    total_diff = 0
    affected_models = []

    for model_name, publisher in dup_models:
        # 获取这个模型的所有记录
        cursor.execute(f"""
            SELECT rowid, download_count, fetched_at
            FROM {DATA_TABLE}
            WHERE date = '2026-01-16'
            AND repo = 'AI Studio'
            AND model_name = ?
            AND publisher = ?
            ORDER BY rowid ASC
        """, (model_name, publisher))
        records = cursor.fetchall()

        print(f"\n{'='*80}")
        print(f"模型: {model_name} (发布者: {publisher})")
        print(f"{'='*80}")
        print(f"{'rowid':<10} {'download_count':<20} {'fetched_at':<30}")
        print("-"*80)

        first_count = None
        last_count = None
        for rowid, count, fetched_at in records:
            # 转换count为整数
            try:
                count_int = int(count)
            except (ValueError, TypeError):
                count_int = 0

            print(f"{rowid:<10} {count:<20} {(fetched_at or 'N/A'):<30}")

            if first_count is None:
                first_count = count_int
            last_count = count_int

        # 计算差异
        if first_count is not None and last_count is not None:
            diff = last_count - first_count
            if diff != 0:
                print(f"\n⚠️ 下载量变化: {first_count:,} → {last_count:,} (差异: {diff:+,})")
                total_diff += diff
                affected_models.append((model_name, first_count, last_count, diff))
            else:
                print(f"\n✅ 下载量无变化")

    # 总结
    print(f"\n{'='*80}")
    print("总结：所有受影响的模型")
    print(f"{'='*80}")
    print(f"{'模型名称':<50} {'原始值':>15} {'新值':>15} {'差异':>15}")
    print("-"*80)

    for model_name, first, last, diff in affected_models:
        print(f"{model_name:<50} {first:>15,} {last:>15,} {diff:>15+,}")

    print(f"\n总差异: {total_diff:,} ({total_diff / 10000:.2f}万)")
    print(f"受影响模型数: {len(affected_models)}")

    # 估算如果用原始值会得到多少
    print(f"\n{'='*80}")
    print("估算：如果用原始值（first_count），1.16总量会减少 ~{:.2f}万".format(total_diff / 10000))
    print(f"{'='*80}")

    conn.close()

if __name__ == '__main__':
    main()
