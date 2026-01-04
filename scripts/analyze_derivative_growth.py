"""
分析最后两个日期的衍生模型新增情况
"""
import sqlite3
import pandas as pd
from collections import defaultdict

DB_PATH = "../ernie_downloads.db"

def analyze_derivative_growth():
    """分析衍生模型增长"""
    conn = sqlite3.connect(DB_PATH)

    # 获取最后两个有数据的日期
    dates_df = pd.read_sql_query(
        "SELECT DISTINCT date FROM model_downloads ORDER BY date DESC LIMIT 2",
        conn
    )

    if len(dates_df) < 2:
        print("数据不足，需要至少两个日期的数据")
        return

    latest_date = dates_df.iloc[0]['date']
    previous_date = dates_df.iloc[1]['date']

    print("=" * 80)
    print(f"对比日期: {previous_date} vs {latest_date}")
    print("=" * 80)

    # 获取两个日期的数据（只看 Hugging Face 平台，因为 model_category 主要在那里）
    query = """
    SELECT date, repo, model_name, publisher, model_category, model_type, base_model
    FROM model_downloads
    WHERE date IN (?, ?) AND repo = 'Hugging Face' AND model_category IS NOT NULL
    """

    df = pd.read_sql_query(query, conn, params=(previous_date, latest_date))
    conn.close()

    print(f"\n总记录数: {len(df)}")
    print(f"  - {previous_date}: {len(df[df['date'] == previous_date])} 条")
    print(f"  - {latest_date}: {len(df[df['date'] == latest_date])} 条")

    # 分别统计两个日期的数据
    def get_stats(date_df):
        """获取某个日期的统计数据"""
        stats = {}
        for category in ['ernie-4.5', 'paddleocr-vl']:
            cat_df = date_df[date_df['model_category'] == category]
            stats[category] = {
                'total': len(cat_df),
                'by_type': cat_df['model_type'].value_counts().to_dict(),
                'models': set(cat_df['model_name'].unique())
            }
        return stats

    prev_stats = get_stats(df[df['date'] == previous_date])
    latest_stats = get_stats(df[df['date'] == latest_date])

    # 打印详细对比
    for category in ['ernie-4.5', 'paddleocr-vl']:
        print(f"\n{'=' * 80}")
        print(f"{category.upper()} 模型分析")
        print(f"{'=' * 80}")

        prev_total = prev_stats[category]['total']
        latest_total = latest_stats[category]['total']
        diff = latest_total - prev_total

        print(f"\n总数变化: {prev_total} → {latest_total} (新增 {diff:+d})")

        # 按类型统计
        print(f"\n按模型类型统计:")
        all_types = set(prev_stats[category]['by_type'].keys()) | set(latest_stats[category]['by_type'].keys())

        for model_type in sorted(all_types):
            prev_count = prev_stats[category]['by_type'].get(model_type, 0)
            latest_count = latest_stats[category]['by_type'].get(model_type, 0)
            type_diff = latest_count - prev_count

            if type_diff != 0:
                print(f"  {model_type:15s}: {prev_count:4d} → {latest_count:4d} ({type_diff:+4d})")
            else:
                print(f"  {model_type:15s}: {prev_count:4d} → {latest_count:4d} (无变化)")

        # 找出新增的模型
        new_models = latest_stats[category]['models'] - prev_stats[category]['models']
        removed_models = prev_stats[category]['models'] - latest_stats[category]['models']

        if new_models:
            print(f"\n新增模型 ({len(new_models)} 个):")
            # 获取新增模型的详细信息
            new_models_df = df[(df['date'] == latest_date) &
                              (df['model_category'] == category) &
                              (df['model_name'].isin(new_models))]

            for _, row in new_models_df.iterrows():
                model_type = row['model_type'] or 'unknown'
                base = row['base_model'] or 'N/A'
                print(f"  + [{model_type:10s}] {row['model_name']}")
                if base != 'N/A' and len(base) < 60:
                    print(f"      └─ base: {base}")

        if removed_models:
            print(f"\n消失的模型 ({len(removed_models)} 个):")
            for model in sorted(removed_models):
                print(f"  - {model}")

if __name__ == "__main__":
    analyze_derivative_growth()
