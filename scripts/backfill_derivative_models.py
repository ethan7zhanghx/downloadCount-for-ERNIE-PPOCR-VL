#!/usr/bin/env python3
"""
回填衍生模型数据到创建日期

对于2026-01-16获取的AI Studio和ModelScope的新增衍生模型（此前未获取过的），
优先使用created_at，备选last_modified，将数据记录回填到模型的创建日期。

避免统计当周新增时记录过多的模型。
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ernie_tracker.config import DB_PATH, DATA_TABLE


def backfill_derivative_models():
    """回填衍生模型数据到创建日期"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    target_date = '2026-01-16'
    target_repos = ['AI Studio', 'ModelScope']
    target_categories = ['ernie-4.5', 'paddleocr-vl']

    print("🚀 开始回填衍生模型数据...")
    print(f"目标日期: {target_date}")
    print(f"目标平台: {', '.join(target_repos)}")
    print(f"目标分类: {', '.join(target_categories)}")
    print()

    # Step 1: 首先运行 backfill_model_category 确保 model_category 字段已填充
    print("Step 1: 检查并填充 model_category 字段...")
    try:
        from scripts.backfill_model_category import backfill_model_category as fill_category
        fill_category()
        print("✅ model_category 字段检查完成\n")
    except Exception as e:
        print(f"⚠️  填充 model_category 时出错: {e}")
        print("继续执行...\n")

    # Step 2: 获取在 target_date 之前的所有 (repo, model_name) 组合
    print("Step 2: 识别历史上已有的模型...")
    cursor.execute(f"""
        SELECT DISTINCT repo, model_name
        FROM {DATA_TABLE}
        WHERE date < ?
        AND repo IN ({','.join(['?' for _ in target_repos])})
    """, [target_date] + target_repos)

    historical_models = set()
    for row in cursor.fetchall():
        historical_models.add((row[0], row[1]))

    print(f"✅ 发现 {len(historical_models)} 个历史模型\n")

    # Step 3: 获取 target_date 的所有衍生模型记录（去重）
    print("Step 3: 获取目标日期的衍生模型...")
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
        GROUP BY repo, model_name
    """, [target_date] + target_repos + target_categories)

    target_records = cursor.fetchall()
    print(f"✅ 目标日期有 {len(target_records)} 个相关模型\n")

    # Step 4: 识别新增的衍生模型
    print("Step 4: 识别新增的衍生模型...")
    new_derivative_models = []

    for record in target_records:
        rowid = record[0]
        repo = record[1]
        model_name = record[2]

        # 检查是否为新增模型（不在历史记录中）
        if (repo, model_name) not in historical_models:
            new_derivative_models.append(record)

    print(f"✅ 发现 {len(new_derivative_models)} 个新增衍生模型\n")

    if len(new_derivative_models) == 0:
        print("❌ 没有发现需要回填的新增衍生模型")
        conn.close()
        return

    # Step 5: 确定回填日期并插入记录
    print("Step 5: 开始回填数据...")
    backfilled_count = 0
    skipped_count = 0

    for record in new_derivative_models:
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

        # 确定回填日期：优先使用 created_at，备选 last_modified
        backfill_date = created_at if created_at else last_modified

        if not backfill_date:
            print(f"⚠️  跳过 {repo} - {model_name}: 没有创建时间或更新时间")
            skipped_count += 1
            continue

        # 确保回填日期在目标日期之前
        if backfill_date >= target_date:
            print(f"⚠️  跳过 {repo} - {model_name}: 创建时间 {backfill_date} 不在目标日期之前")
            skipped_count += 1
            continue

        # 检查该日期是否已有记录
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM {DATA_TABLE}
            WHERE date = ? AND repo = ? AND model_name = ?
        """, [backfill_date, repo, model_name])

        if cursor.fetchone()[0] > 0:
            print(f"⚠️  跳过 {repo} - {model_name}: 日期 {backfill_date} 已有记录")
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

    # 验证结果
    print("\n" + "="*60)
    print("✅ 回填完成！")
    print("="*60)
    print(f"成功回填: {backfilled_count} 条记录")
    print(f"跳过记录: {skipped_count} 条")

    # 显示按平台统计的结果
    print("\n" + "="*60)
    print("按平台统计回填情况:")
    print("="*60)

    for repo in target_repos:
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM {DATA_TABLE}
            WHERE date < ? AND repo = ? AND download_count = '0'
        """, [target_date, repo])

        zero_count = cursor.fetchone()[0]
        print(f"  {repo:20s}: {zero_count} 条回填记录")

    conn.close()

    print("\n✅ 所有操作已完成")


if __name__ == '__main__':
    print("="*60)
    print("衍生模型数据回填工具")
    print("="*60)
    print()

    try:
        backfill_derivative_models()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
