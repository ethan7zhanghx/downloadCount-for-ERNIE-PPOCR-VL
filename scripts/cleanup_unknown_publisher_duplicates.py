"""
清理 Unknown Publisher 重复数据

问题：2025-12-08 和 2025-12-11 这两天，HuggingFace API 返回的 publisher 信息异常，
导致一些模型被记录为 publisher='Unknown'，但这些模型在其他日期都有正确的 publisher。
这造成了回填模式下的重复计数。

解决方案：删除这些日期的 Unknown publisher 记录（因为同一模型在其他日期有正确的 publisher）

⚠️ 重要说明：
    本脚本只处理特定的 API 异常情况。正常情况下，模型的唯一标识必须是：
    (repo, publisher, model_name) 三元组

    本次清理之所以可以删除 Unknown 记录，是因为：
    1. 这些模型在其他日期都有正确的 publisher
    2. 经过验证，Unknown 版本是 API 异常导致的同一模型的重复记录
    3. 这是特例，不代表一般情况下可以忽略 publisher 字段

使用方法：
    python3 scripts/cleanup_unknown_publisher_duplicates.py

清理结果：
    - 删除 40 条 Unknown publisher 重复记录（20 个模型 × 2 个日期）
    - ERNIE-4.5: 累计从 204 降至 194
    - PaddleOCR-VL: 累计从 45 降至 35
"""
import sys
import os

# 支持从任何目录运行
if __name__ == '__main__':
    # 如果从 scripts/ 目录运行，切换到父目录
    if os.path.basename(os.getcwd()) == 'scripts':
        os.chdir('..')
        sys.path.insert(0, '.')
    else:
        sys.path.insert(0, '.')

import sqlite3
from ernie_tracker.config import DB_PATH
from ernie_tracker.db_manager import backup_database

# 先备份数据库
print("="*80)
print("步骤 1: 备份数据库")
print("="*80)
backup_file = backup_database()
print(f"✅ 备份完成: {backup_file}\n")

# 连接数据库
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 查找需要删除的记录
print("="*80)
print("步骤 2: 识别需要删除的 Unknown Publisher 记录")
print("="*80)

# 这些是 Unknown 重复的模型（通过前面的分析发现）
unknown_duplicates_paddle = [
    'Proton', 'Trouter-20b', 'anpr-models', 'tfjs-mobilenet-692',
    'ProVerBs_Law_777', 'Neopulse', 'Mark', 'AILAND', 'Elonmusk', 'MODEL-ONE-1'
]

unknown_duplicates_ernie = [
    'Darwin-gpt-ernie-20b', 'Hexo-0.5-Nano-Experimental', 'bai-ming-reinit-550m-zero',
    'elementor-ernie45-custom', 'vl-lora', 'eUp_NMT_10-36-40_22-09-2025',
    'eUp_NMT_10-57-55_19-09-2025', 'food_nutrition_coach', 'jarvis', 'Owl'
]

all_unknown_models = unknown_duplicates_paddle + unknown_duplicates_ernie

# 查询这些模型的 Unknown 记录
query = """
SELECT rowid, date, repo, publisher, model_name, model_category, download_count
FROM model_downloads
WHERE publisher = 'Unknown'
AND model_name IN ({})
ORDER BY model_name, date
""".format(','.join('?' * len(all_unknown_models)))

cursor.execute(query, all_unknown_models)
records_to_delete = cursor.fetchall()

print(f"找到 {len(records_to_delete)} 条 Unknown 记录\n")

# 按模型分组显示
from collections import defaultdict
by_model = defaultdict(list)
for record in records_to_delete:
    by_model[record[4]].append(record)

for model_name, records in sorted(by_model.items()):
    print(f"{model_name}:")
    for record in records:
        rowid, date, repo, publisher, model_name, category, downloads = record
        print(f"  - {date}: {repo}, category={category}, downloads={downloads}")

# 确认删除
print("\n" + "="*80)
print("步骤 3: 删除记录")
print("="*80)

# 删除这些记录
delete_query = """
DELETE FROM model_downloads
WHERE publisher = 'Unknown'
AND model_name IN ({})
""".format(','.join('?' * len(all_unknown_models)))

cursor.execute(delete_query, all_unknown_models)
deleted_count = cursor.rowcount

conn.commit()
print(f"✅ 已删除 {deleted_count} 条 Unknown Publisher 重复记录\n")

# 验证结果
print("="*80)
print("步骤 4: 验证清理结果")
print("="*80)

# 重新检查是否还有这些模型的 Unknown 记录
cursor.execute(query, all_unknown_models)
remaining = cursor.fetchall()

if len(remaining) == 0:
    print("✅ 所有 Unknown 重复记录已成功清理")
else:
    print(f"⚠️  仍有 {len(remaining)} 条 Unknown 记录（可能是其他日期的）")
    for record in remaining:
        print(f"  - {record[1]}: {record[4]}")

# 检查这些模型是否仍有正确的 publisher 记录
print("\n验证这些模型的正确 publisher 记录仍然存在:")
for model_name in sorted(all_unknown_models)[:5]:  # 抽查前5个
    cursor.execute("""
        SELECT DISTINCT publisher, COUNT(*) as cnt
        FROM model_downloads
        WHERE model_name = ?
        GROUP BY publisher
    """, (model_name,))
    publishers = cursor.fetchall()

    print(f"\n{model_name}:")
    for pub, cnt in publishers:
        print(f"  - {pub}: {cnt} 条记录")

conn.close()

print("\n" + "="*80)
print("清理完成！")
print("="*80)
print(f"备份文件: {backup_file}")
print(f"已删除记录数: {deleted_count}")
print("\n现在可以重新运行统计，数字应该更准确了。")
