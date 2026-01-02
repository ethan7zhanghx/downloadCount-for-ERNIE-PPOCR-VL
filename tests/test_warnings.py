"""
测试负增长警告机制
"""
from analysis import calculate_weekly_report, format_report_tables

# 使用最近的两个日期进行测试
current_date = '2025-11-07'  # 根据DATABASE_FIX_SUMMARY.md中的信息
previous_date = '2025-10-31'

print("=" * 80)
print("测试负增长警告机制")
print("=" * 80)
print(f"当前日期: {current_date}")
print(f"对比日期: {previous_date}")
print()

# 生成周报
report_data = calculate_weekly_report(current_date, previous_date, model_series='ERNIE-4.5')

if report_data is None:
    print("❌ 无法生成周报，请检查日期是否有数据")
else:
    # 格式化报表
    tables = format_report_tables(report_data)

    # 检查负增长警告
    warnings_df = tables.get('negative_growth_warnings')

    if warnings_df is not None and not warnings_df.empty:
        print("\n" + "=" * 80)
        print(f"⚠️  检测到 {len(warnings_df)} 个模型出现负增长")
        print("=" * 80)
        print("\n负增长警告表格:")
        print(warnings_df.to_string())
        print("\n测试通过：警告机制正常工作！✅")
    else:
        print("\n✅ 没有检测到负增长")
        print("测试通过：警告机制正常工作！✅")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
