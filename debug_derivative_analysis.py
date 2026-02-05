#!/usr/bin/env python3
"""调试衍生模型生态分析"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ernie_tracker.db import load_data_from_db
from ernie_tracker.analysis import normalize_model_names, mark_official_models
import pandas as pd

def main():
    print("="*80)
    print("调试Gitee平台衍生模型分析")
    print("="*80)

    # 加载全量数据
    df = load_data_from_db(date_filter=None, last_value_per_model=False)
    print(f"\n1. 加载数据: {len(df)} 条记录")

    # 筛选Gitee平台
    gitee_df = df[df['repo'] == 'Gitee'].copy()
    print(f"\n2. Gitee平台: {len(gitee_df)} 条记录")

    # 标准化publisher名称
    gitee_df['publisher'] = gitee_df['publisher'].astype(str).apply(
        lambda x: x.title() if x.lower() != 'nan' else x
    )
    print(f"\n3. 标准化publisher后的Gitee记录: {len(gitee_df)}")

    # 检查unique (repo, publisher, model_name) 组合数
    unique_combos = gitee_df.drop_duplicates(subset=['repo', 'publisher', 'model_name'])
    print(f"\n4. 唯一(repo, publisher, model_name)组合: {len(unique_combos)}")

    # 标准化模型名称
    gitee_df = normalize_model_names(gitee_df)
    print(f"\n5. 标准化model_name后的Gitee记录: {len(gitee_df)}")

    # 检查标准化后的unique组合
    unique_combos_after = gitee_df.drop_duplicates(subset=['repo', 'publisher', 'model_name'])
    print(f"\n6. 标准化后唯一(repo, publisher, model_name)组合: {len(unique_combos_after)}")

    # 按日期筛选 (<= 2026-01-24)
    gitee_df['date'] = pd.to_datetime(gitee_df['date'])
    gitee_df = gitee_df[gitee_df['date'] <= '2026-01-24'].copy()
    print(f"\n7. 筛选日期<=1.24后的Gitee记录: {len(gitee_df)}")

    # 标记官方模型
    gitee_df = mark_official_models(gitee_df)
    print(f"\n8. 标记官方模型后的Gitee记录: {len(gitee_df)}")
    print(f"   - 官方模型: {(gitee_df['is_official'] == True).sum()}")
    print(f"   - 衍生模型: {(gitee_df['is_official'] == False).sum()}")

    # 按model_category筛选 (ernie-4.5)
    gitee_ernie = gitee_df[
        (gitee_df['model_category'] == 'ernie-4.5') |
        (gitee_df['model_name'].str.contains('ERNIE-4.5', case=False, na=False))
    ].copy()
    print(f"\n9. 筛选ernie-4.5后的Gitee记录: {len(gitee_ernie)}")
    print(f"   - 官方模型: {(gitee_ernie['is_official'] == True).sum()}")
    print(f"   - 衍生模型: {(gitee_ernie['is_official'] == False).sum()}")

    # 去重 (取历史最大值)
    gitee_ernie = gitee_ernie.sort_values('download_count', ascending=False).drop_duplicates(
        subset=['repo', 'publisher', 'model_name'], keep='first'
    )
    print(f"\n10. 去重后的Gitee记录: {len(gitee_ernie)}")
    print(f"   - 官方模型: {(gitee_ernie['is_official'] == True).sum()}")
    print(f"   - 衍生模型: {(gitee_ernie['is_official'] == False).sum()}")

    # 检查重复的模型
    print(f"\n11. 检查可能的重复模型:")
    model_groups = gitee_ernie.groupby('model_name').size()
    duplicate_models = model_groups[model_groups > 1]
    if len(duplicate_models) > 0:
        print(f"   发现 {len(duplicate_models)} 个重复模型:")
        for model_name, count in duplicate_models.head(10).items():
            print(f"   - {model_name}: {count}条")
            # 显示这些重复记录的publisher
            records = gitee_ernie[gitee_ernie['model_name'] == model_name][['publisher', 'is_official', 'download_count', 'date']]
            for _, row in records.iterrows():
                print(f"     {row['publisher']} | official={row['is_official']} | {row['download_count']} | {row['date']}")
    else:
        print(f"   ✅ 没有重复模型")

    # 显示所有衍生模型
    print(f"\n12. Gitee衍生模型列表:")
    derivative_df = gitee_ernie[gitee_ernie['is_official'] == False]
    print(f"   共 {len(derivative_df)} 个衍生模型")
    for _, row in derivative_df.iterrows():
        print(f"   - {row['model_name']} (publisher={row['publisher']}, downloads={row['download_count']})")

if __name__ == '__main__':
    main()
