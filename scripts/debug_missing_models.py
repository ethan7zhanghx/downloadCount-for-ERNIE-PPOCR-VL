"""
调试：找出累计统计中丢失的2个模型
"""
import sys
sys.path.insert(0, '..')

from ernie_tracker.db import load_data_from_db
from ernie_tracker.analysis import get_deleted_or_hidden_models
import pandas as pd

# 1. 当天有数据的衍生模型（正常模式）
current_actual = load_data_from_db(date_filter='2026-01-02', last_value_per_model=False)
current_derivatives = current_actual[
    (current_actual['model_category'] == 'ernie-4.5') &
    (current_actual['model_type'] != 'original')
].copy()

# 2. 已删除模型
deleted_models = get_deleted_or_hidden_models('2026-01-02', model_series='ERNIE-4.5')

# 3. 回填模式的所有模型
all_historical = load_data_from_db(date_filter='2026-01-02', last_value_per_model=True)
historical_derivatives = all_historical[
    (all_historical['model_category'] == 'ernie-4.5') &
    (all_historical['model_type'] != 'original')
].copy()

print(f'回填模式总数: {len(historical_derivatives)}')
print(f'当天模式总数: {len(current_derivatives)}')
print(f'已删除模型: {len(deleted_models)}')

# 创建唯一标识
current_derivatives['key'] = current_derivatives['repo'] + '|||' + current_derivatives['publisher'] + '|||' + current_derivatives['model_name']
historical_derivatives['key'] = historical_derivatives['repo'] + '|||' + historical_derivatives['publisher'] + '|||' + historical_derivatives['model_name']

current_keys = set(current_derivatives['key'].unique())
historical_keys = set(historical_derivatives['key'].unique())

print(f'\n当天唯一key数量: {len(current_keys)}')
print(f'回填唯一key数量: {len(historical_keys)}')

# 找出只在回填中有，不在当天的（这应该是已删除的）
deleted_keys = historical_keys - current_keys
print(f'差集（已删除）: {len(deleted_keys)}')

# 找出在当天和回填都没有的
deleted_model_keys = set()
for m in deleted_models:
    key = m['repo'] + '|||' + m['publisher'] + '|||' + m['model_name']
    deleted_model_keys.add(key)

print(f'已删除模型key数量: {len(deleted_model_keys)}')

# 找出差异
only_in_deleted = deleted_keys - deleted_model_keys
only_in_function = deleted_model_keys - deleted_keys

if only_in_deleted:
    print(f'\n只在差集中，不在函数返回的已删除列表: {len(only_in_deleted)}')
    for key in sorted(only_in_deleted):
        parts = key.split('|||')
        print(f'  平台: {parts[0]} | Publisher: {parts[1]} | 模型: {parts[2]}')

if only_in_function:
    print(f'\n只在函数返回，不在差集中: {len(only_in_function)}')
    for key in sorted(only_in_function):
        parts = key.split('|||')
        print(f'  平台: {parts[0]} | Publisher: {parts[1]} | 模型: {parts[2]}')

# 现在应用和周报相同的标准化逻辑
print('\n' + '='*80)
print('应用周报的标准化逻辑后:')
print('='*80)

from ernie_tracker.analysis import normalize_model_names

def enforce_deduplication_and_standardization(df):
    if df.empty:
        return df

    # 1. 标准化 publisher 名称
    df['publisher'] = df['publisher'].astype(str).apply(lambda x: x.title() if x.lower() != 'nan' else x)

    # 2. 标准化模型名称
    df = normalize_model_names(df)

    # 3. 再次去重
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0)
    df = df.sort_values(by='download_count', ascending=False).drop_duplicates(
        subset=['date', 'repo', 'publisher', 'model_name'], keep='first'
    )
    return df

# 应用标准化
historical_normalized = enforce_deduplication_and_standardization(all_historical.copy())
historical_normalized = historical_normalized[
    (historical_normalized['model_category'] == 'ernie-4.5') &
    (historical_normalized['model_type'] != 'original')
]

print(f'标准化后回填模式总数: {len(historical_normalized)}')
