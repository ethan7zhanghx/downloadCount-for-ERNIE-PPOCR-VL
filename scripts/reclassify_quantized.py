"""
重新分类数据库中的量化模型
将之前被分类为 'other' 的量化模型重新标记为 'quantized'
"""
import sqlite3
import pandas as pd
from ernie_tracker.fetchers.fetchers_modeltree import classify_model_type
from ernie_tracker.config import DB_PATH


def reclassify_quantized_models(dry_run=True):
    """
    重新分类数据库中的量化模型

    Args:
        dry_run: 如果为 True，只显示会被修改的记录，不实际修改数据库
    """
    conn = sqlite3.connect(DB_PATH)

    try:
        # 读取所有 model_type 为 'other' 的记录
        query = """
        SELECT rowid, date, repo, model_name, publisher, download_count,
               model_type, model_category, tags, base_model, data_source
        FROM model_downloads
        WHERE model_type = 'other' OR model_type IS NULL
        """

        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("✅ 没有需要重新分类的记录")
            return

        print(f"📊 共找到 {len(df)} 条 model_type='other' 或 NULL 的记录")

        # 重新分类
        reclassified_count = 0
        reclassified_records = []

        for idx, row in df.iterrows():
            model_name = row['model_name']
            publisher = row['publisher']
            full_model_id = f"{publisher}/{model_name}"

            # 解析 tags（如果有的话）
            tags = []
            if pd.notna(row['tags']) and row['tags']:
                try:
                    tags = eval(row['tags']) if isinstance(row['tags'], str) else row['tags']
                except:
                    tags = []

            # 重新分类
            new_type = classify_model_type(full_model_id, tags, None)

            # 如果新分类为 quantized，记录下来
            if new_type == 'quantized' and row['model_type'] != 'quantized':
                reclassified_count += 1
                reclassified_records.append({
                    'rowid': row['rowid'],
                    'model_name': model_name,
                    'publisher': publisher,
                    'old_type': row['model_type'],
                    'new_type': new_type,
                    'date': row['date'],
                    'repo': row['repo']
                })

        if reclassified_count == 0:
            print("✅ 没有需要重新分类为 'quantized' 的记录")
            return

        print(f"\n🔄 发现 {reclassified_count} 个模型需要重新分类为 'quantized':")

        # 显示前20个
        display_df = pd.DataFrame(reclassified_records[:20])
        print("\n前20个需要重新分类的模型:")
        print(display_df[['model_name', 'publisher', 'old_type', 'new_type', 'date']].to_string(index=False))

        if len(reclassified_records) > 20:
            print(f"\n... 还有 {len(reclassified_records) - 20} 个模型未显示")

        if dry_run:
            print("\n⚠️ DRY RUN 模式：不会实际修改数据库")
            print("如需实际执行，请运行: python reclassify_quantized.py --execute")
        else:
            print("\n⏳ 开始更新数据库...")
            cursor = conn.cursor()

            for record in reclassified_records:
                cursor.execute(
                    "UPDATE model_downloads SET model_type = ? WHERE rowid = ?",
                    (record['new_type'], record['rowid'])
                )

            conn.commit()
            print(f"✅ 成功更新 {reclassified_count} 条记录")

            # 显示更新后的统计
            print("\n📊 更新后的 model_type 统计:")
            stats_query = """
            SELECT model_type, COUNT(*) as count
            FROM model_downloads
            WHERE model_type IS NOT NULL AND model_type != ''
            GROUP BY model_type
            ORDER BY count DESC
            """
            stats_df = pd.read_sql_query(stats_query, conn)
            print(stats_df.to_string(index=False))

    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    # 检查命令行参数
    execute = '--execute' in sys.argv or '-e' in sys.argv
    force = '--force' in sys.argv or '-f' in sys.argv

    if execute:
        print("🚀 执行模式：将实际修改数据库")
        if force:
            print("⚡ 强制执行模式：跳过确认")
            reclassify_quantized_models(dry_run=False)
        else:
            try:
                confirm = input("确认执行？(yes/no): ")
                if confirm.lower() in ['yes', 'y']:
                    reclassify_quantized_models(dry_run=False)
                else:
                    print("❌ 已取消")
            except EOFError:
                print("\n❌ 无法获取用户输入，已取消")
                print("提示：如需非交互式执行，请使用 --force 参数")
    else:
        print("🔍 DRY RUN 模式：仅预览，不会修改数据库\n")
        reclassify_quantized_models(dry_run=True)
