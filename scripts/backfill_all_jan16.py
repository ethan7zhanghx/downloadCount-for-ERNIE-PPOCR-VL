#!/usr/bin/env python3
"""
回填1月16日所有衍生模型到其last_modified日期

对于2026-01-16获取的所有AI Studio和ModelScope衍生模型，
只要last_modified早于2026-01-16，就回填到last_modified日期。
"""

import sqlite3
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ernie_tracker.config import DB_PATH, DATA_TABLE


def backfill_all_jan16_models():
    """回填1月16日所有衍生模型"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    target_date = '2026-01-16'
    target_repos = ['AI Studio', 'ModelScope']
    target_categories = ['ernie-4.5', 'paddleocr-vl']
    target_types = ['finetune', 'quantized', 'adapter', 'lora', 'merge']

    print("🚀 开始回填1月16日所有衍生模型...")
    print(f"目标日期: {target_date}")
    print()

    # 获取1月16日的所有衍生模型（去重）
    print("Step 1: 获取1月16日的所有衍生模型...")
    cursor.execute(f"""
        SELECT
            MAX(rowid) as rowid,
            repo,
            model_name,
            publisher,
            download_count,
            model_type,
            model_category,
            tags,
            base_model,
            data_source,
            likes,
            library_name,
            pipeline_tag,
            created_at,
            last_modified,
            fetched_at,
            base_model_from_api,
            search_keyword,
            url
        FROM {DATA_TABLE}
        WHERE date = ?
        AND repo IN ({','.join(['?' for _ in target_repos])})
        AND model_category IN ({','.join(['?' for _ in target_categories])})
        AND model_type IN ({','.join(['?' for _ in target_types])})
        GROUP BY repo, model_name
    """, [target_date] + target_repos + target_categories + target_types)

    target_records = cursor.fetchall()
    print(f"✅ 找到 {len(target_records)} 个衍生模型\n")

    # 开始回填
    print("Step 2: 开始回填...")
    backfilled_count = 0
    skipped_count = 0

    for record in target_records:
        (
            rowid,
            repo,
            model_name,
            publisher,
            download_count,
            model_type,
            model_category,
            tags,
            base_model,
            data_source,
            likes,
            library_name,
            pipeline_tag,
            created_at,
            last_modified,
            fetched_at,
            base_model_from_api,
            search_keyword,
            url
        ) = record

        # 确定回填日期：优先使用 last_modified，备选 created_at
        backfill_date = last_modified if last_modified else created_at

        if not backfill_date:
            skipped_count += 1
            continue

        # 只回填早于目标日期的记录
        if backfill_date >= target_date:
            skipped_count += 1
            continue

        # 检查该日期是否已有记录
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM {DATA_TABLE}
            WHERE date = ? AND repo = ? AND model_name = ?
        """, [backfill_date, repo, model_name])

        if cursor.fetchone()[0] > 0:
            skipped_count += 1
            continue

        # 插入回填记录（下载量为0）
        cursor.execute(f"""
            INSERT INTO {DATA_TABLE} (
                date, repo, model_name, publisher, download_count,
                model_type, model_category, tags, base_model, data_source,
                likes, library_name, pipeline_tag,
                created_at, last_modified, fetched_at,
                base_model_from_api, search_keyword, url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            backfill_date, repo, model_name, publisher, '0',
            model_type, model_category, tags, base_model, data_source,
            likes, library_name, pipeline_tag,
            created_at, last_modified, fetched_at,
            base_model_from_api, search_keyword, url
        ])

        backfilled_count += 1
        print(f"✅ 回填: {repo} - {model_name} -> {backfill_date}")

    # 提交事务
    conn.commit()

    # 显示结果
    print("\n" + "="*60)
    print("✅ 回填完成！")
    print("="*60)
    print(f"成功回填: {backfilled_count} 条记录")
    print(f"跳过记录: {skipped_count} 条")

    # 按平台统计回填情况
    print("\n" + "="*60)
    print("按平台统计回填情况:")
    print("="*60)

    for repo in target_repos:
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM {DATA_TABLE}
            WHERE date < ?
            AND repo = ?
            AND download_count = '0'
            AND model_category IN ({','.join(['?' for _ in target_categories])})
        """, [target_date, repo] + target_categories)

        zero_count = cursor.fetchone()[0]
        print(f"  {repo:20s}: {zero_count} 条回填记录")

    conn.close()
    print("\n✅ 所有操作已完成")


if __name__ == '__main__':
    print("="*60)
    print("批量回填1月16日衍生模型工具")
    print("="*60)
    print()

    try:
        backfill_all_jan16_models()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
