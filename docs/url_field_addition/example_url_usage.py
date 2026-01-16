#!/usr/bin/env python3
"""
示例：使用URL字段导出模型列表及其详情页链接
"""
import pandas as pd
from ernie_tracker.fetchers.fetchers_unified import fetch_hugging_face_data_unified

def main():
    print("=" * 80)
    print("示例：获取HuggingFace模型及其URL")
    print("=" * 80)

    # 获取数据（使用快速搜索模式）
    df, count = fetch_hugging_face_data_unified(use_model_tree=False)

    print(f"\n获取到 {count} 个模型")

    # 显示前10个模型的URL
    print("\n前10个模型的URL示例：")
    print("-" * 80)

    for idx, row in df.head(10).iterrows():
        print(f"\n模型: {row['model_name']}")
        print(f"发布者: {row['publisher']}")
        print(f"下载量: {row['download_count']:,}")
        if 'url' in row and pd.notna(row['url']):
            print(f"URL: {row['url']}")
        else:
            print("URL: (无)")

    # 导出到CSV
    output_file = "model_urls_example.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n{'=' * 80}")
    print(f"✅ 已导出到: {output_file}")
    print(f"{'=' * 80}")

    # 统计URL覆盖情况
    has_url = df['url'].notna().sum()
    print(f"\nURL统计:")
    print(f"  总模型数: {len(df)}")
    print(f"  有URL的模型: {has_url}")
    print(f"  覆盖率: {has_url/len(df)*100:.1f}%")

if __name__ == "__main__":
    main()
