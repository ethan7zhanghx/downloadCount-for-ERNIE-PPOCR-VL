import pandas as pd
import sqlite3
import sys
from datetime import date

# 假设数据库路径在 config.py 中定义，但为了独立运行，我们硬编码或从命令行获取
DB_PATH = "ernie_downloads.db" # 假设数据库文件名为 ernie_downloads.db

def import_excel_to_db(excel_path):
    """
    读取 Excel 文件中的数据并导入到 SQLite 数据库的 model_downloads 表中。
    """
    try:
        # 读取 Excel 文件。由于我们之前发现它只有一个 Sheet1，我们直接读取
        df = pd.read_excel(excel_path, sheet_name=0)
    except FileNotFoundError:
        print(f"错误: 文件未找到: {excel_path}")
        return
    except Exception as e:
        print(f"读取 Excel 文件时发生错误: {e}")
        return

    # 假设 Excel 文件的列名与数据库表 model_downloads 的列名一致:
    # date, repo, model_name, publisher, download_count
    
    # 检查关键列是否存在
    required_cols = ['date', 'repo', 'model_name', 'publisher', 'download_count']
    if not all(col in df.columns for col in required_cols):
        print(f"错误: Excel 文件缺少必需的列。必需列: {required_cols}")
        print(f"文件中的列: {df.columns.tolist()}")
        return

    # 确保 download_count 是数值类型，并处理可能的非数值项
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)
    
    # 确保 date 列是字符串格式
    df['date'] = df['date'].astype(str)

    conn = sqlite3.connect(DB_PATH)
    
    try:
        # 插入数据到 model_downloads 表
        # 使用 replace 模式，如果存在相同的 (date, repo, model_name) 组合，则替换
        df.to_sql("model_downloads", conn, if_exists="append", index=False)
        print(f"成功从 {excel_path} 导入 {len(df)} 条记录到数据库 {DB_PATH}。")
    except Exception as e:
        print(f"导入数据到数据库时发生错误: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        import_excel_to_db(sys.argv[1])
    else:
        print("用法: python3 import_excel.py <excel_file_path>")
