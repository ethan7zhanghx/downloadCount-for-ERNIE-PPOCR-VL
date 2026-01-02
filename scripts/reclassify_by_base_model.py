"""
根据 base_model 重新分类数据库中的模型系列
确保 Model Tree 衍生模型的 model_category 正确反映其 base_model 所属系列
"""
import sqlite3
import pandas as pd
from ernie_tracker.fetchers.fetchers_modeltree import classify_model
from ernie_tracker.config import DB_PATH


def reclassify_by_base_model(dry_run=True):
    """
    根据 base_model 重新分类模型系列

    Args:
        dry_run: 如果为 True，只显示会被修改的记录，不实际修改数据库
    """
    conn = sqlite3.connect(DB_PATH)

    try:
        # 读取所有有 base_model 的记录（Model Tree 衍生模型）
        query = """
        SELECT rowid, date, repo, model_name, publisher, download_count,
               model_type, model_category, tags, base_model, data_source
        FROM model_downloads
        WHERE base_model IS NOT NULL
          AND base_model != ''
          AND base_model != 'None'
          AND repo = 'Hugging Face'
        """

        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("✅ 没有需要重新分类的 Model Tree 衍生模型")
            return

        print(f"📊 共找到 {len(df)} 条 Model Tree 衍生模型记录")

        # 重新分类
        reclassified_count = 0
        reclassified_records = []

        for idx, row in df.iterrows():
            model_name = row['model_name']
            publisher = row['publisher']
            full_model_id = f"{publisher}/{model_name}"
            base_model = row['base_model']
            old_category = row['model_category']

            # 重新分类（使用 base_model）
            new_category = classify_model(full_model_id, publisher, base_model=base_model)

            # 如果分类发生变化，记录下来
            if new_category != old_category:
                reclassified_count += 1
                reclassified_records.append({
                    'rowid': row['rowid'],
                    'model_name': model_name,
                    'publisher': publisher,
                    'base_model': base_model,
                    'old_category': old_category,
                    'new_category': new_category,
                    'date': row['date']
                })

        if reclassified_count == 0:
            print("✅ 所有 Model Tree 衍生模型的分类都是正确的")
            return

        print(f"\n🔄 发现 {reclassified_count} 个模型需要重新分类:")

        # 按新分类分组统计
        reclassified_df = pd.DataFrame(reclassified_records)
        category_changes = reclassified_df.groupby(['old_category', 'new_category']).size().reset_index(name='count')

        print("\n分类变化汇总:")
        for _, row in category_changes.iterrows():
            print(f"  {row['old_category']} → {row['new_category']}: {row['count']} 个")

        # 显示前20个
        print("\n前20个需要重新分类的模型:")
        display_df = reclassified_df.head(20)[['model_name', 'base_model', 'old_category', 'new_category']]
        print(display_df.to_string(index=False))

        if len(reclassified_records) > 20:
            print(f"\n... 还有 {len(reclassified_records) - 20} 个模型未显示")

        if dry_run:
            print("\n⚠️ DRY RUN 模式：不会实际修改数据库")
            print("如需实际执行，请运行: python reclassify_by_base_model.py --execute --force")
        else:
            print("\n⏳ 开始更新数据库...")
            cursor = conn.cursor()

            for record in reclassified_records:
                cursor.execute(
                    "UPDATE model_downloads SET model_category = ? WHERE rowid = ?",
                    (record['new_category'], record['rowid'])
                )

            conn.commit()
            print(f"✅ 成功更新 {reclassified_count} 条记录")

            # 显示更新后的统计
            print("\n📊 更新后的 model_category 统计:")
            stats_query = """
            SELECT model_category, COUNT(*) as count
            FROM model_downloads
            WHERE model_category IS NOT NULL AND model_category != ''
            GROUP BY model_category
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
            reclassify_by_base_model(dry_run=False)
        else:
            try:
                confirm = input("确认执行？(yes/no): ")
                if confirm.lower() in ['yes', 'y']:
                    reclassify_by_base_model(dry_run=False)
                else:
                    print("❌ 已取消")
            except EOFError:
                print("\n❌ 无法获取用户输入，已取消")
                print("提示：如需非交互式执行，请使用 --force 参数")
    else:
        print("🔍 DRY RUN 模式：仅预览，不会修改数据库\n")
        reclassify_by_base_model(dry_run=True)
