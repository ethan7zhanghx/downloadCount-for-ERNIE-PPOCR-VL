import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ernie_tracker.fetchers.fetchers import fetch_gitcode_data

if __name__ == "__main__":
    print("开始获取 GitCode 数据...")
    df, total_count = fetch_gitcode_data()
    print(f"获取到 {total_count} 条 GitCode 数据。")
    print(df.head())
