import pandas as pd
import sqlite3
import sys

DB_PATH = "ernie_downloads.db"
OUTPUT_CSV_PATH = "db_export.csv"

def export_db_to_csv():
    """
    从 SQLite 数据库的 model_downloads 表中读取所有数据，并将其导出到 CSV 文件。
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM model_downloads", conn)
        conn.close()
        
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"成功将 {len(df)} 条记录从数据库 {DB_PATH} 导出到 {OUTPUT_CSV_PATH}。")
        
    except Exception as e:
        print(f"导出数据时发生错误: {e}")

if __name__ == "__main__":
    export_db_to_csv()
