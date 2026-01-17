#!/usr/bin/env python3
"""
修正数据库中包含K/M后缀的下载量记录

将 "7.3k" 转换为 "7300"，将 "1.2M" 转换为 "1200000" 等
"""
import sqlite3
import re
from ernie_tracker.config import DB_PATH, DATA_TABLE


def convert_download_count(value):
    """
    将 K/M 格式的下载量转换为纯数字字符串

    Args:
        value: 下载量值（可能是 "7.3k", "1.2M", 或 "1234"）

    Returns:
        转换后的字符串（如 "7300", "1200000", "1234"）
    """
    if not value or pd.isna(value):
        return value

    value_str = str(value).strip().lower()

    # 如果已经是纯数字，直接返回
    if value_str.isdigit():
        return value_str

    # 匹配 K 后缀（如 7.3k, 1.2k, 1k）
    k_match = re.match(r'^(\d+(?:\.\d+)?)k$', value_str)
    if k_match:
        num = float(k_match.group(1))
        return str(int(num * 1000))

    # 匹配 M 后缀（如 1.2M, 1M）
    m_match = re.match(r'^(\d+(?:\.\d+)?)m$', value_str)
    if m_match:
        num = float(m_match.group(1))
        return str(int(num * 1000000))

    # 如果不匹配任何模式，返回原值
    return value


def fix_download_counts():
    """修正数据库中所有包含K/M后缀的下载量记录"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 查询所有记录
    cursor.execute(f"""
        SELECT rowid, date, repo, model_name, publisher, download_count,
               model_type, model_category, tags, base_model, data_source,
               likes, library_name, pipeline_tag, created_at, last_modified,
               fetched_at, base_model_from_api, search_keyword
        FROM {DATA_TABLE}
        WHERE download_count LIKE '%K%' OR download_count LIKE '%M%'
           OR download_count LIKE '%k%' OR download_count LIKE '%m%'
    """)

    records = cursor.fetchall()

    if not records:
        print("未找到包含K/M后缀的下载量记录")
        conn.close()
        return

    print(f"找到 {len(records)} 条需要修正的记录")

    # 备份数据
    print("\n原始记录：")
    for record in records:
        rowid = record[0]
        download_count = record[4]
        model_name = record[3]
        print(f"  [{rowid}] {model_name}: {download_count}")

    # 修正每条记录
    updated = 0
    for record in records:
        rowid = record[0]
        download_count = record[4]

        # 转换下载量格式
        new_count = convert_download_count(download_count)

        if new_count != download_count:
            cursor.execute(
                f"UPDATE {DATA_TABLE} SET download_count = ? WHERE rowid = ?",
                (new_count, rowid)
            )
            updated += 1
            print(f"  [{rowid}] {download_count} -> {new_count}")

    conn.commit()
    conn.close()

    print(f"\n成功修正 {updated} 条记录")


if __name__ == "__main__":
    import pandas as pd
    fix_download_counts()
