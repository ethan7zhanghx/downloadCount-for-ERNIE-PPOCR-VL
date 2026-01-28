#!/usr/bin/env python3
"""
补齐 Hugging Face 的 created_at, last_modified, tags, pipeline_tag, library_name, likes 字段

这些字段在不带 expand 的 API 调用中可以获取，但历史数据中可能缺失。
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ernie_tracker.config import DB_PATH, DATA_TABLE

try:
    from huggingface_hub import model_info
except ImportError:
    print("请先安装 huggingface_hub: pip3 install huggingface_hub")
    sys.exit(1)


def get_hf_model_info(model_id):
    """
    调用 Hugging Face API（不带expand）获取模型信息

    返回字典，包含:
    - created_at
    - last_modified
    - tags
    - pipeline_tag
    - library_name
    - likes
    """
    try:
        info = model_info(model_id)  # 不带 expand

        # 处理 tags (list -> JSON string)
        tags = getattr(info, 'tags', None)
        if tags and isinstance(tags, list):
            import json
            tags = json.dumps(tags)
        elif tags is None:
            tags = '[]'

        # 处理日期
        created_at = getattr(info, 'created_at', None)
        if created_at:
            if isinstance(created_at, datetime):
                created_at = created_at.strftime('%Y-%m-%d')
            else:
                created_at = str(created_at)[:10]

        last_modified = getattr(info, 'last_modified', None)
        if last_modified:
            if isinstance(last_modified, datetime):
                last_modified = last_modified.strftime('%Y-%m-%d')
            else:
                last_modified = str(last_modified)[:10]

        return {
            'created_at': created_at,
            'last_modified': last_modified,
            'tags': tags,
            'pipeline_tag': getattr(info, 'pipeline_tag', None),
            'library_name': getattr(info, 'library_name', None),
            'likes': getattr(info, 'likes', None),
        }
    except Exception as e:
        print(f"  ⚠️  获取 {model_id} 失败: {e}")
        return None


def backfill_hf_fields(conn):
    """补齐 Hugging Face 的字段"""
    cursor = conn.cursor()

    # 1. 找出需要补齐的模型 (Hugging Face 且字段缺失)
    cursor.execute(f"""
        SELECT DISTINCT publisher, model_name
        FROM {DATA_TABLE}
        WHERE repo = 'Hugging Face'
          AND (
            created_at IS NULL OR created_at = ''
            OR last_modified IS NULL OR last_modified = ''
            OR tags IS NULL OR tags = ''
            OR pipeline_tag IS NULL OR pipeline_tag = ''
            OR library_name IS NULL OR library_name = ''
            OR likes IS NULL OR likes = ''
          )
    """)
    models_to_fetch = cursor.fetchall()

    if not models_to_fetch:
        print("✅ 无需补齐，所有 Hugging Face 记录字段完整")
        return 0

    print(f"📋 需要补齐的模型数: {len(models_to_fetch)}")
    print()

    # 2. 调用 API 获取字段
    model_data = {}
    for i, (publisher, model_name) in enumerate(models_to_fetch, 1):
        model_id = f"{publisher}/{model_name}"
        print(f"[{i}/{len(models_to_fetch)}] 获取 {model_id}...")

        data = get_hf_model_info(model_id)
        if data:
            model_data[(publisher, model_name)] = data

    print()
    print(f"✅ 成功获取 {len(model_data)} 个模型的信息")
    print()

    # 3. 更新数据库
    if not model_data:
        print("⚠️  没有获取到任何数据，取消更新")
        return 0

    total_updated = 0
    for (publisher, model_name), data in model_data.items():
        # 构建更新语句（只更新空值字段）
        updates = []
        params = []

        if data['created_at']:
            updates.append("created_at = ?")
            params.append(data['created_at'])
        if data['last_modified']:
            updates.append("last_modified = ?")
            params.append(data['last_modified'])
        if data['tags']:
            updates.append("tags = ?")
            params.append(data['tags'])
        if data['pipeline_tag']:
            updates.append("pipeline_tag = ?")
            params.append(data['pipeline_tag'])
        if data['library_name']:
            updates.append("library_name = ?")
            params.append(data['library_name'])
        if data['likes'] is not None:
            updates.append("likes = ?")
            params.append(str(data['likes']))

        if updates:
            params.extend([publisher, model_name])
            sql = f"""
                UPDATE {DATA_TABLE}
                SET {', '.join(updates)}
                WHERE repo = 'Hugging Face'
                  AND publisher = ?
                  AND model_name = ?
            """
            cursor.execute(sql, params)
            total_updated += cursor.rowcount

    conn.commit()
    print(f"✅ 更新完成，共影响 {total_updated} 条记录")

    return total_updated


def verify_coverage(conn):
    """验证更新后的覆盖率"""
    cursor = conn.cursor()

    print()
    print("="*60)
    print("更新后 Hugging Face 字段覆盖率:")
    print("-"*60)

    cursor.execute(f"""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN created_at IS NOT NULL AND created_at != '' THEN 1 END) as created_at,
            COUNT(CASE WHEN last_modified IS NOT NULL AND last_modified != '' THEN 1 END) as last_modified,
            COUNT(CASE WHEN tags IS NOT NULL AND tags != '' THEN 1 END) as tags,
            COUNT(CASE WHEN pipeline_tag IS NOT NULL AND pipeline_tag != '' THEN 1 END) as pipeline_tag,
            COUNT(CASE WHEN library_name IS NOT NULL AND library_name != '' THEN 1 END) as library_name,
            COUNT(CASE WHEN likes IS NOT NULL AND likes != '' THEN 1 END) as likes
        FROM {DATA_TABLE}
        WHERE repo = 'Hugging Face'
    """)

    row = cursor.fetchone()
    total = row[0]

    fields = ['created_at', 'last_modified', 'tags', 'pipeline_tag', 'library_name', 'likes']
    for i, field in enumerate(fields, 1):
        count = row[i]
        coverage = (count / total * 100) if total > 0 else 0
        print(f"  {field:20s}: {count:5d} / {total:5d} ({coverage:5.1f}%)")


def main():
    print("="*60)
    print("补齐 Hugging Face 字段 (通过不带expand的API)")
    print("="*60)
    print(f"数据库: {DB_PATH}")
    print()

    conn = sqlite3.connect(DB_PATH)

    try:
        updated = backfill_hf_fields(conn)
        if updated > 0:
            verify_coverage(conn)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()


if __name__ == '__main__':
    main()
