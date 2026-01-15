#!/usr/bin/env python3
"""
补齐数据库中的 model_category 字段（通用版本）

可以指定数据库路径，适用于任何位置的数据库
"""

import sqlite3
import sys
import argparse
from pathlib import Path

# 默认数据表名
DATA_TABLE = "model_downloads"


def classify_model_category(model_name, search_keyword):
    """
    根据 search_keyword 和模型名称判断模型分类

    Args:
        model_name: 模型名称
        search_keyword: 搜索关键词（可能为 NULL）

    Returns:
        str: model_category ('ernie-4.5', 'paddleocr-vl', 'other-ernie', 'other')
    """
    model_name = str(model_name).lower()

    # 1. 优先使用 search_keyword
    if search_keyword and search_keyword != 'None':
        search_keyword = str(search_keyword).upper()
        if 'ERNIE-4.5' in search_keyword or search_keyword == 'ERNIE-4.5':
            return 'ernie-4.5'
        elif 'PADDLEOCR-VL' in search_keyword or search_keyword == 'PADDLEOCR-VL':
            return 'paddleocr-vl'

    # 2. 使用模型名称判断
    if 'ernie-4.5' in model_name or 'ernie4.5' in model_name or ('文心' in model_name and '4.5' in model_name):
        return 'ernie-4.5'
    elif 'paddleocr-vl' in model_name or 'paddleocrvl' in model_name:
        return 'paddleocr-vl'
    elif 'ernie' in model_name or '文心' in model_name:
        return 'other-ernie'
    else:
        return 'other'


def backfill_model_category(db_path, dry_run=False):
    """
    补齐 model_category 字段

    Args:
        db_path: 数据库文件路径
        dry_run: 是否只显示不执行更新
    """
    if not Path(db_path).exists():
        print(f"❌ 数据库文件不存在: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{DATA_TABLE}'")
    if not cursor.fetchone():
        print(f"❌ 数据库中没有 {DATA_TABLE} 表")
        conn.close()
        return

    # 统计需要更新的记录数
    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {DATA_TABLE}
        WHERE model_category IS NULL
           OR model_category = ''
           OR LOWER(model_category) = 'none'
           OR LOWER(model_category) = 'nan'
    """)
    total_to_update = cursor.fetchone()[0]

    if total_to_update == 0:
        print("✅ 所有记录的 model_category 字段都已填充，无需更新")
        conn.close()
        return

    print(f"📊 数据库: {db_path}")
    print(f"📊 发现 {total_to_update:,} 条记录需要更新 model_category 字段")

    if dry_run:
        print("\n🔍 预览模式 - 显示前10条需要更新的记录:")
        cursor.execute(f"""
            SELECT date, repo, model_name, publisher, search_keyword
            FROM {DATA_TABLE}
            WHERE model_category IS NULL
               OR model_category = ''
               OR LOWER(model_category) = 'none'
               OR LOWER(model_category) = 'nan'
            ORDER BY date DESC
            LIMIT 10
        """)
        rows = cursor.fetchall()
        for row in rows:
            category = classify_model_category(row[2], row[4])
            print(f"  {row[0]} | {row[1]:15s} | {row[2]:30s} | {row[3]:20s} -> {category}")
        print(f"\n... 还有 {total_to_update - 10} 条记录")
        conn.close()
        return

    print("开始处理...")

    # 获取需要更新的记录（rowid, model_name, search_keyword）
    cursor.execute(f"""
        SELECT rowid, model_name, search_keyword
        FROM {DATA_TABLE}
        WHERE model_category IS NULL
           OR model_category = ''
           OR LOWER(model_category) = 'none'
           OR LOWER(model_category) = 'nan'
    """)

    records_to_update = cursor.fetchall()

    # 统计分类结果
    category_counts = {
        'ernie-4.5': 0,
        'paddleocr-vl': 0,
        'other-ernie': 0,
        'other': 0
    }

    # 批量更新
    updates = []
    for rowid, model_name, search_keyword in records_to_update:
        category = classify_model_category(model_name, search_keyword)
        updates.append((category, rowid))
        category_counts[category] += 1

    # 执行更新
    cursor.executemany(f"""
        UPDATE {DATA_TABLE}
        SET model_category = ?
        WHERE rowid = ?
    """, updates)

    conn.commit()

    # 验证更新结果
    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {DATA_TABLE}
        WHERE model_category IS NULL
           OR model_category = ''
           OR LOWER(model_category) = 'none'
           OR LOWER(model_category) = 'nan'
    """)
    remaining = cursor.fetchone()[0]

    conn.close()

    # 打印结果
    print("\n" + "="*60)
    print("✅ 更新完成！")
    print("="*60)
    print(f"总更新记录数: {total_to_update - remaining:,}")
    print(f"\n分类统计:")
    print(f"  - ernie-4.5:     {category_counts['ernie-4.5']:,} 条")
    print(f"  - paddleocr-vl:  {category_counts['paddleocr-vl']:,} 条")
    print(f"  - other-ernie:   {category_counts['other-ernie']:,} 条")
    print(f"  - other:         {category_counts['other']:,} 条")

    if remaining > 0:
        print(f"\n⚠️ 仍有 {remaining} 条记录未更新")
    else:
        print("\n✅ 所有记录已成功更新")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='补齐数据库中的 model_category 字段')
    parser.add_argument(
        '--db-path',
        default='ernie_downloads.db',
        help='数据库文件路径 (默认: ernie_downloads.db)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，只显示不执行更新'
    )

    args = parser.parse_args()

    print("🚀 开始补齐 model_category 字段...")
    print()

    try:
        backfill_model_category(args.db_path, args.dry_run)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
