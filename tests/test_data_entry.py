"""测试数据录入功能"""
from ernie_tracker.db_manager import insert_single_record, import_from_excel
import pandas as pd
from datetime import date
import os

print("=" * 80)
print("测试数据录入功能")
print("=" * 80)

# 测试 1: 单条数据录入
print("\n【测试 1】单条数据录入")
print("-" * 80)

success, message = insert_single_record(
    date='2025-01-15',
    repo='Hugging Face',
    model_name='测试模型_单条录入',
    publisher='测试发布者',
    download_count=1234,
    base_model=None,
    model_type='finetune',
    model_category='ernie-4.5'
)

if success:
    print(f"✅ {message}")
else:
    print(f"❌ {message}")

# 测试 2: Excel 批量导入
print("\n【测试 2】Excel 批量导入")
print("-" * 80)

# 创建测试数据
test_data = {
    'date': ['2025-01-15', '2025-01-15', '2025-01-15'],
    'repo': ['Hugging Face', 'ModelScope', 'AI Studio'],
    'model_name': ['测试模型_批量1', '测试模型_批量2', '测试模型_批量3'],
    'publisher': ['测试发布者A', '测试发布者B', '测试发布者C'],
    'download_count': [100, 200, 300],
    'base_model': ['ERNIE-4.5-21B-A3B-PT', '', ''],
    'model_type': ['finetune', 'adapter', ''],
    'model_category': ['ernie-4.5', 'ernie-4.5', '']
}

test_df = pd.DataFrame(test_data)
test_file = 'test_import.xlsx'

# 保存为 Excel
test_df.to_excel(test_file, index=False, engine='openpyxl')
print(f"📄 创建测试文件: {test_file}")

# 导入测试
success, message, stats = import_from_excel(test_file, skip_duplicates=True)

if success:
    print(f"✅ 导入成功！")
    print(f"   - 总记录数: {stats['total']}")
    print(f"   - 成功插入: {stats['inserted']}")
    print(f"   - 跳过重复: {stats['skipped']}")
    print(f"   - 错误记录: {stats['errors']}")
else:
    print(f"❌ 导入失败")
    print(f"   {message}")

# 清理测试文件
if os.path.exists(test_file):
    os.remove(test_file)
    print(f"\n🧹 已清理测试文件")

# 测试 3: 重复记录检测
print("\n【测试 3】重复记录检测")
print("-" * 80)

success2, message2 = insert_single_record(
    date='2025-01-15',
    repo='Hugging Face',
    model_name='测试模型_单条录入',
    publisher='测试发布者',
    download_count=5678
)

if not success2:
    print(f"✅ 正确检测到重复记录")
    print(f"   {message2}")
else:
    print(f"❌ 应该检测到重复但没有: {message2}")

# 测试 4: 数据验证
print("\n【测试 4】数据验证")
print("-" * 80)

# 测试负数下载量
success3, message3 = insert_single_record(
    date='2025-01-15',
    repo='Hugging Face',
    model_name='测试模型_负数',
    publisher='测试发布者',
    download_count=-100
)

if not success3:
    print(f"✅ 正确拒绝负数下载量")
    print(f"   {message3}")
else:
    print(f"❌ 应该拒绝负数但没有: {message3}")

# 测试无效日期格式
success4, message4 = insert_single_record(
    date='invalid-date',
    repo='Hugging Face',
    model_name='测试模型_无效日期',
    publisher='测试发布者',
    download_count=100
)

if not success4:
    print(f"✅ 正确拒绝无效日期格式")
    print(f"   {message4}")
else:
    print(f"❌ 应该拒绝无效日期但没有: {message4}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
print("\n💡 提示：测试数据已录入数据库，可以在 Streamlit UI 中查看")
print("   运行: streamlit run app.py")
print("   然后进入「数据库管理」->「数据录入」页面查看")
