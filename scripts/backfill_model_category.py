#!/usr/bin/env python3
"""
补齐数据库中的 model_category 字段

策略：
1. 优先使用 search_keyword 字段判断
2. 如果没有 search_keyword，使用模型名称判断
3. 只更新 model_category 为 NULL 或空字符串的记录
"""

import sqlite3
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ernie_tracker.config import DB_PATH, DATA_TABLE


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
    if search_keyword:
        search_keyword = str(search_keyword).upper()
        if 'ERNIE-4.5' in search_keyword or search_keyword == 'ERNIE-4.5':
            return 'ernie-4.5'
        elif 'PADDLEOCR-VL' in search_keyword or search_keyword == 'PaddleOCR-VL':
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


def backfill_model_category():
    """补齐 model_category 字段"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 统计需要更新的记录数
    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {DATA_TABLE}
        WHERE model_category IS NULL OR model_category = ''
    """)
    total_to_update = cursor.fetchone()[0]

    if total_to_update == 0:
        print("✅ 所有记录的 model_category 字段都已填充，无需更新")
        conn.close()
        return

    print(f"📊 发现 {total_to_update} 条记录需要更新 model_category 字段")
    print("开始处理...")

    # 获取需要更新的记录（rowid, model_name, search_keyword）
    cursor.execute(f"""
        SELECT rowid, model_name, search_keyword
        FROM {DATA_TABLE}
        WHERE model_category IS NULL OR model_category = ''
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
        WHERE model_category IS NULL OR model_category = ''
    """)
    remaining = cursor.fetchone()[0]

    conn.close()

    # 打印结果
    print("\n" + "="*60)
    print("✅ 更新完成！")
    print("="*60)
    print(f"总更新记录数: {total_to_update - remaining}")
    print(f"\n分类统计:")
    print(f"  - ernie-4.5:     {category_counts['ernie-4.5']:,} 条")
    print(f"  - paddleocr-vl:  {category_counts['paddleocr-vl']:,} 条")
    print(f"  - other-ernie:   {category_counts['other-ernie']:,} 条")
    print(f"  - other:         {category_counts['other']:,} 条")

    if remaining > 0:
        print(f"\n⚠️ 仍有 {remaining} 条记录未更新")
    else:
        print("\n✅ 所有记录已成功更新")

    # 显示按平台统计的结果
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n" + "="*60)
    print("按平台统计 model_category 填充情况:")
    print("="*60)

    cursor.execute(f"""
        SELECT
            repo,
            COUNT(*) as total,
            SUM(CASE WHEN model_category IS NOT NULL AND model_category != '' THEN 1 ELSE 0 END) as with_category
        FROM {DATA_TABLE}
        GROUP BY repo
        ORDER BY total DESC
    """)

    for repo, total, with_category in cursor.fetchall():
        percentage = (with_category / total * 100) if total > 0 else 0
        print(f"  {repo:20s}: {with_category:4d}/{total:4d} ({percentage:5.1f}%)")

    conn.close()


if __name__ == '__main__':
    print("🚀 开始补齐 model_category 字段...")
    print()

    try:
        backfill_model_category()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
