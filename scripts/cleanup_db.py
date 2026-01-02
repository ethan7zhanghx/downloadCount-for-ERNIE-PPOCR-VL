import pandas as pd
import sqlite3

DB_PATH = "ernie_downloads.db"

def cleanup_model_names():
    """
    清理数据库中 model_downloads 表的 model_name 字段。
    对于 repo 为 'AI Studio' 或 'ModelScope' 的记录，
    如果 model_name 包含 '/'，则将其拆分，更新 model_name 和 publisher。
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM model_downloads", conn)
        
        if df.empty:
            print("数据库为空，无需清理。")
            return

        print(f"开始清理 {len(df)} 条记录...")
        
        cleaned_count = 0
        
        # 创建一个新的 DataFrame 来存储清理后的数据
        cleaned_df = df.copy()

        for index, row in cleaned_df.iterrows():
            if row['repo'] in ['AI Studio', 'ModelScope', 'Gitee'] and isinstance(row['model_name'], str) and '/' in row['model_name']:
                parts = row['model_name'].split('/', 1)
                publisher = parts[0]
                model_name = parts[1]
                
                # 更新 DataFrame 中的值
                cleaned_df.at[index, 'publisher'] = publisher
                cleaned_df.at[index, 'model_name'] = model_name
                cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"共清理了 {cleaned_count} 条记录。")
            # 使用 'replace' 模式将整个清理后的 DataFrame 写回数据库，覆盖旧表
            cleaned_df.to_sql("model_downloads", conn, if_exists='replace', index=False)
            print("数据库已成功更新。")
        else:
            print("未发现需要清理的记录。")

    except Exception as e:
        print(f"清理数据库时发生错误: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    cleanup_model_names()
