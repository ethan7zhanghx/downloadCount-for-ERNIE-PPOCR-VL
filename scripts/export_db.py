import pandas as pd
import sqlite3
import sys
from datetime import date
from pathlib import Path

# 使用统一的数据目录
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "ernie_downloads.db"
EXPORTS_DIR = DATA_DIR / "exports"

def export_db_to_excel():
    """
    从 SQLite 数据库的 model_downloads 表中读取所有数据，并将其导出到 Excel 文件。
    导出到 data/exports/ 目录
    """
    try:
        # 确保导出目录存在
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # 生成带日期的文件名
        output_file = EXPORTS_DIR / f"database_export_{date.today().isoformat()}.xlsx"

        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("SELECT * FROM model_downloads", conn)
        conn.close()

        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✅ 成功将 {len(df)} 条记录导出到 {output_file}")

    except Exception as e:
        print(f"❌ 导出数据时发生错误: {e}")

def export_db_to_csv():
    """
    从 SQLite 数据库导出到 CSV（备用格式）
    """
    try:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = EXPORTS_DIR / f"database_export_{date.today().isoformat()}.csv"

        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("SELECT * FROM model_downloads", conn)
        conn.close()

        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 成功将 {len(df)} 条记录导出到 {output_file}")

    except Exception as e:
        print(f"❌ 导出数据时发生错误: {e}")

if __name__ == "__main__":
    export_db_to_csv()
