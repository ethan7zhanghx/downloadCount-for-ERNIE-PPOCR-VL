"""
修复数据库中 model tree 模型的 tags 和分类
重新从 HuggingFace 获取 tags 并重新分类
"""
import sqlite3
import pandas as pd
from huggingface_hub import model_info
from ernie_tracker.fetchers.fetchers_modeltree import classify_model_type
from ernie_tracker.config import DB_PATH, DATA_TABLE


def fix_model_tree_tags():
    """修复数据库中有 base_model 但 tags 为空的模型"""

    conn = sqlite3.connect(DB_PATH)

    # 查找需要修复的模型（所有有 base_model 的模型都重新分类）
    # 使用 date, repo, model_name 作为唯一标识，而不是 ROWID
    query = f"""
        SELECT date, repo, model_name, publisher, base_model, tags, model_type
        FROM {DATA_TABLE}
        WHERE base_model IS NOT NULL
        AND base_model != ''
        AND base_model != 'None'
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        print("✅ 没有需要修复的模型")
        conn.close()
        return

    print(f"🔧 找到 {len(df)} 个需要修复的模型\n")

    fixed_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        model_name = row['model_name']
        publisher = row['publisher']
        full_id = f"{publisher}/{model_name}"

        try:
            # 如果 tags 为空，从 HuggingFace 重新获取
            if not row['tags'] or row['tags'] in ['None', '', '[]']:
                print(f"  📥 获取 {full_id} 的详细信息...")
                try:
                    info = model_info(full_id)
                    tags = getattr(info, 'tags', [])
                    tags_str = str(tags) if tags else '[]'
                except Exception as e:
                    print(f"    ⚠️ 获取失败: {e}")
                    error_count += 1
                    continue
            else:
                # 使用已有的 tags
                print(f"  🔄 重新分类 {full_id}...")
                tags_str = row['tags']
                try:
                    import ast
                    tags = ast.literal_eval(tags_str)
                except:
                    tags = []

            # 重新分类
            new_type = classify_model_type(
                full_id,
                tags,
                None
            )

            # 只在分类变化时更新
            if new_type != row['model_type']:
                # 更新数据库（使用 date, repo, model_name 作为唯一标识）
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    UPDATE {DATA_TABLE}
                    SET tags = ?, model_type = ?
                    WHERE date = ? AND repo = ? AND model_name = ?
                    """,
                    (tags_str, new_type, row['date'], row['repo'], row['model_name'])
                )

                print(f"    ✅ {full_id}: {row['model_type']} -> {new_type}")
                fixed_count += 1
            else:
                print(f"    ⏭️  {full_id}: {row['model_type']} (无变化)")


        except Exception as e:
            print(f"    ❌ {full_id} 修复失败: {e}")
            error_count += 1

    conn.commit()
    conn.close()

    print(f"\n📊 修复完成:")
    print(f"  ✅ 成功修复: {fixed_count} 个")
    print(f"  ❌ 失败: {error_count} 个")

    # 显示修复后的统计
    conn = sqlite3.connect(DB_PATH)
    stats_df = pd.read_sql(
        f"""
        SELECT model_type, COUNT(*) as count
        FROM {DATA_TABLE}
        WHERE base_model IS NOT NULL
        AND base_model != ''
        AND base_model != 'None'
        GROUP BY model_type
        """,
        conn
    )
    conn.close()

    print(f"\n📈 修复后 Model Tree 模型分类统计:")
    for _, row in stats_df.iterrows():
        print(f"  {row['model_type']}: {row['count']} 个")


if __name__ == "__main__":
    print("=== 修复 Model Tree 模型的 Tags 和分类 ===\n")
    fix_model_tree_tags()
