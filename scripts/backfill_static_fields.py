#!/usr/bin/env python3
"""
回填非时变字段（model_type, base_model, tags, created_at）

对于每个 (repo, publisher, model_name) 组合，如果有任何记录包含这些字段的值，
则将该值回填到所有空值的历史记录中。
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ernie_tracker.config import DB_PATH, DATA_TABLE


def backfill_field(conn, field_name):
    """
    回填单个字段

    对于每个 (repo, publisher, model_name) 组合：
    1. 找出该组合下该字段有值的记录
    2. 将该值回填到所有空值记录
    """
    cursor = conn.cursor()

    # 检查需要回填的记录数
    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {DATA_TABLE}
        WHERE {field_name} IS NULL OR {field_name} = ''
    """)
    null_count = cursor.fetchone()[0]

    if null_count == 0:
        print(f"  ✅ {field_name}: 无需回填（已完整）")
        return 0

    # 统计可回填数量
    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {DATA_TABLE} m
        WHERE m.{field_name} IS NULL OR m.{field_name} = ''
          AND EXISTS (
            SELECT 1 FROM {DATA_TABLE} src
            WHERE src.repo = m.repo
              AND src.publisher = m.publisher
              AND src.model_name = m.model_name
              AND src.{field_name} IS NOT NULL
              AND src.{field_name} != ''
          )
    """)
    can_backfill = cursor.fetchone()[0]

    if can_backfill == 0:
        print(f"  ⚠️  {field_name}: 有 {null_count} 条空值，但无可回填来源")
        return 0

    # 执行回填
    print(f"  🔄 {field_name}: 回填中... (可回填 {can_backfill} 条)")

    # SQLite 不支持在 UPDATE 中直接引用同一个表，使用分步方式
    # 1. 获取所有需要回填的值
    cursor.execute(f"""
        SELECT DISTINCT
            m.repo, m.publisher, m.model_name,
            (SELECT src.{field_name}
             FROM {DATA_TABLE} src
             WHERE src.repo = m.repo
               AND src.publisher = m.publisher
               AND src.model_name = m.model_name
               AND src.{field_name} IS NOT NULL
               AND src.{field_name} != ''
             LIMIT 1) as value_to_fill
        FROM {DATA_TABLE} m
        WHERE m.{field_name} IS NULL OR m.{field_name} = ''
          AND EXISTS (
            SELECT 1 FROM {DATA_TABLE} src
            WHERE src.repo = m.repo
              AND src.publisher = m.publisher
              AND src.model_name = m.model_name
              AND src.{field_name} IS NOT NULL
              AND src.{field_name} != ''
          )
    """)

    # 2. 逐条更新
    updates = []
    for repo, publisher, model_name, value in cursor.fetchall():
        updates.append((value, repo, publisher, model_name))

    # 3. 批量执行更新
    for value, repo, publisher, model_name in updates:
        cursor.execute(f"""
            UPDATE {DATA_TABLE}
            SET {field_name} = ?
            WHERE repo = ? AND publisher = ? AND model_name = ?
              AND ({field_name} IS NULL OR {field_name} = '')
        """, (value, repo, publisher, model_name))

    updated = cursor.rowcount
    print(f"  ✅ {field_name}: 已回填 {updated} 条记录")

    return updated


def verify_backfill(conn, field_name):
    """验证回填后的覆盖率"""
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN {field_name} IS NOT NULL AND {field_name} != '' THEN 1 END) as has_value
        FROM {DATA_TABLE}
    """)
    total, has_value = cursor.fetchone()
    coverage = (has_value / total * 100) if total > 0 else 0

    return total, has_value, coverage


def main():
    print("="*60)
    print("回填非时变字段")
    print("="*60)
    print(f"数据库: {DB_PATH}")
    print()

    # 显示回填前状态
    print("回填前状态:")
    print("-"*60)
    conn = sqlite3.connect(DB_PATH)

    fields = ['model_type', 'base_model', 'tags', 'created_at']

    for field in fields:
        total, has_value, coverage = verify_backfill(conn, field)
        print(f"  {field:20s}: {has_value:5d} / {total:5d} ({coverage:5.1f}%)")

    print()
    print("="*60)
    print("开始回填...")
    print("="*60)
    print()

    total_updated = 0
    for field in fields:
        updated = backfill_field(conn, field)
        total_updated += updated
        print()

    # 提交更改
    conn.commit()

    # 显示回填后状态
    print("="*60)
    print("回填后状态:")
    print("-"*60)

    for field in fields:
        total, has_value, coverage = verify_backfill(conn, field)
        print(f"  {field:20s}: {has_value:5d} / {total:5d} ({coverage:5.1f}%)")

    print()
    print(f"✅ 回填完成！共更新 {total_updated} 条记录")

    conn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
