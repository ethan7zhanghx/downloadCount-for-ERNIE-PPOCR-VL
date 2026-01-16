#!/usr/bin/env python3
"""
清理重复的回填记录

对于同一平台、同一模型、同一日期的多条记录，只保留一条（最早插入的，即rowid最小的）。
"""

import sqlite3
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ernie_tracker.config import DB_PATH, DATA_TABLE


def cleanup_duplicate_backfill():
    """清理重复的回填记录"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("🚀 开始清理重复的回填记录...")
    print()

    # 查找所有重复的记录（同一 repo, model_name, date 有多条）
    print("Step 1: 查找重复的回填记录...")
    cursor.execute(f"""
        SELECT repo, model_name, date, COUNT(*) as count
        FROM {DATA_TABLE}
        WHERE download_count = '0'
        GROUP BY repo, model_name, date
        HAVING count > 1
        ORDER BY count DESC
    """)

    duplicates = cursor.fetchall()
    print(f"✅ 发现 {len(duplicates)} 组重复记录\n")

    if len(duplicates) == 0:
        print("❌ 没有发现重复记录")
        conn.close()
        return

    # 统计需要删除的记录数
    total_to_delete = 0
    for repo, model_name, date, count in duplicates:
        total_to_delete += (count - 1)

    print(f"预计需要删除 {total_to_delete} 条重复记录\n")

    # 删除重复记录，保留每组中 rowid 最小的
    print("Step 2: 删除重复记录...")
    deleted_count = 0

    for repo, model_name, date, count in duplicates:
        # 获取该组的所有 rowid，按 rowid 排序
        cursor.execute(f"""
            SELECT rowid
            FROM {DATA_TABLE}
            WHERE repo = ? AND model_name = ? AND date = ?
            ORDER BY rowid ASC
        """, [repo, model_name, date])

        rowids = [row[0] for row in cursor.fetchall()]

        # 保留第一个（rowid最小的），删除其余的
        for rowid in rowids[1:]:
            cursor.execute(f"""
                DELETE FROM {DATA_TABLE}
                WHERE rowid = ?
            """, [rowid])
            deleted_count += 1

        print(f"✅ 删除 {repo} - {model_name} ({date}): {count - 1} 条重复记录")

    # 提交事务
    conn.commit()

    # 验证结果
    print("\n" + "="*60)
    print("✅ 清理完成！")
    print("="*60)
    print(f"删除记录数: {deleted_count}")

    # 验证是否还有重复
    cursor.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT repo, model_name, date, COUNT(*) as count
            FROM {DATA_TABLE}
            WHERE download_count = '0'
            GROUP BY repo, model_name, date
            HAVING count > 1
        )
    """)

    remaining_duplicates = cursor.fetchone()[0]

    if remaining_duplicates > 0:
        print(f"⚠️  仍有 {remaining_duplicates} 组重复记录")
    else:
        print("✅ 所有重复记录已清理")

    # 显示清理后的统计
    print("\n" + "="*60)
    print("清理后的回填记录统计:")
    print("="*60)

    cursor.execute(f"""
        SELECT repo, COUNT(*) as count
        FROM {DATA_TABLE}
        WHERE download_count = '0'
        GROUP BY repo
        ORDER BY count DESC
    """)

    for repo, count in cursor.fetchall():
        print(f"  {repo:20s}: {count} 条")

    conn.close()
    print("\n✅ 所有操作已完成")


if __name__ == '__main__':
    print("="*60)
    print("清理重复回填记录工具")
    print("="*60)
    print()

    try:
        cleanup_duplicate_backfill()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
