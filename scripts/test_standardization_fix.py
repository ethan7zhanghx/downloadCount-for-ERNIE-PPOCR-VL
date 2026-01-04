"""
测试数据标准化修复
验证累计、当前、已删除模型数的数学关系是否正确
"""
import sys
sys.path.insert(0, '..')

from ernie_tracker.db import load_data_from_db
from ernie_tracker.analysis import get_deleted_or_hidden_models, normalize_model_names
import pandas as pd

def test_model_series(date, series_name):
    """测试指定模型系列的数学关系"""
    print(f"\n{'='*60}")
    print(f"测试 {series_name} ({date})")
    print('='*60)

    category = 'ernie-4.5' if series_name == 'ERNIE-4.5' else 'paddleocr-vl'

    # 1. 获取累计数据（应用标准化）
    backfill = load_data_from_db(date_filter=date, last_value_per_model=True)

    # 应用标准化逻辑
    backfill['publisher'] = backfill['publisher'].astype(str).apply(
        lambda x: x.title() if x.lower() != 'nan' else x
    )
    backfill = normalize_model_names(backfill)
    backfill['download_count'] = pd.to_numeric(backfill['download_count'], errors='coerce').fillna(0)
    backfill = backfill.sort_values(by='download_count', ascending=False).drop_duplicates(
        subset=['date', 'repo', 'publisher', 'model_name'], keep='first'
    )

    cumulative = backfill[
        (backfill['model_category'] == category) &
        (backfill['model_type'] != 'original')
    ]

    # 2. 获取当前数据
    current = load_data_from_db(date_filter=date, last_value_per_model=False)
    current_derivatives = current[
        (current['model_category'] == category) &
        (current['model_type'] != 'original')
    ]

    # 3. 获取已删除模型
    deleted = get_deleted_or_hidden_models(date, model_series=series_name)

    # 4. 验证数学关系
    cumulative_count = len(cumulative)
    current_count = len(current_derivatives)
    deleted_count = len(deleted)
    expected_cumulative = current_count + deleted_count

    print(f"\n📊 统计结果:")
    print(f"  累计衍生模型（标准化后）: {cumulative_count} 个")
    print(f"  当前日期衍生模型: {current_count} 个")
    print(f"  已删除/隐藏模型: {deleted_count} 个")

    print(f"\n🔢 数学验证:")
    print(f"  {current_count} + {deleted_count} = {expected_cumulative}")

    if expected_cumulative == cumulative_count:
        print(f"  ✅ 正确！等于累计数 {cumulative_count}")
        success = True
    else:
        print(f"  ❌ 错误！不等于累计数 {cumulative_count}")
        print(f"  差异: {abs(expected_cumulative - cumulative_count)} 个模型")
        success = False

    # 5. 显示部分已删除模型
    if deleted_count > 0:
        print(f"\n🗑️  已删除模型示例（前5个）:")
        for model in deleted[:5]:
            print(f"  - {model['publisher']}/{model['model_name']}")
            print(f"    最后出现: {model['last_seen_date']}, 下载量: {model['last_download_count']}")

    return success

def main():
    """运行所有测试"""
    print("="*60)
    print("数据标准化修复验证测试")
    print("="*60)

    test_date = '2026-01-02'

    # 测试 ERNIE-4.5
    result1 = test_model_series(test_date, 'ERNIE-4.5')

    # 测试 PaddleOCR-VL
    result2 = test_model_series(test_date, 'PaddleOCR-VL')

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    if result1 and result2:
        print("✅ 所有测试通过！数学关系正确。")
        return 0
    else:
        print("❌ 部分测试失败！请检查标准化逻辑。")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
