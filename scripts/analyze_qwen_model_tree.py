"""
分析 Qwen Model Tree Excel 数据
生成每个 base model 不同类型衍生模型的统计
"""
import pandas as pd
import sys

def analyze_model_tree(excel_file):
    """
    分析 Model Tree Excel 数据

    Args:
        excel_file: Excel 文件路径
    """
    print(f"📊 读取文件: {excel_file}")
    df = pd.read_excel(excel_file, sheet_name=0)

    print(f"✅ 数据读取成功，共 {len(df)} 行\n")

    # 1. 总体统计
    print("="*80)
    print("📊 总体统计")
    print("="*80)

    total_models = len(df)
    base_models = df[df['is_base'] == True]
    derivative_models = df[df['is_base'] == False]

    print(f"总记录数: {total_models:,}")
    print(f"基础模型数: {len(base_models)}")
    print(f"衍生模型总数: {len(derivative_models):,}")

    # 2. 基础模型列表
    print(f"\n基础模型列表:")
    for idx, row in base_models.iterrows():
        print(f"  {row['model_id']} (下载量: {row['download_count']:,})")

    # 3. 各基础模型的衍生统计
    print(f"\n{'='*80}")
    print(f"📋 各基础模型的衍生模型统计")
    print(f"{'='*80}\n")

    stats_data = []

    for base_model_id in base_models['model_id'].unique():
        # 获取该 base model 的所有衍生模型
        derivatives = df[(df['base_model'] == base_model_id) & (df['is_base'] == False)]

        if len(derivatives) == 0:
            continue

        print(f"{'─'*80}")
        print(f"🎯 {base_model_id}")
        print(f"{'─'*80}")

        # 按类型统计
        type_counts = derivatives['model_type'].value_counts()
        print(f"衍生模型总数: {len(derivatives)}")
        print(f"\n按类型分布:")
        for model_type, count in type_counts.items():
            percentage = (count / len(derivatives)) * 100
            print(f"  {model_type:15s}: {count:4d} ({percentage:5.1f}%)")

        # 下载量统计
        total_deriv_downloads = derivatives['download_count'].sum()
        avg_deriv_downloads = derivatives['download_count'].mean()
        max_deriv = derivatives.loc[derivatives['download_count'].idxmax()]

        print(f"\n下载量统计:")
        print(f"  衍生模型总下载量: {total_deriv_downloads:,}")
        print(f"  平均每个衍生模型: {avg_deriv_downloads:,.0f}")
        print(f"  最受欢迎的衍生模型: {max_deriv['model_id']} ({max_deriv['download_count']:,})")

        # 按发布者统计（Top 5）
        publisher_counts = derivatives['publisher'].value_counts().head(5)
        print(f"\nTop 5 发布者:")
        for publisher, count in publisher_counts.items():
            print(f"  {publisher:30s}: {count} 个模型")

        # 按类型统计下载量
        quantized_downloads = derivatives[derivatives['model_type'] == 'quantized']['download_count'].sum()
        finetune_downloads = derivatives[derivatives['model_type'] == 'finetune']['download_count'].sum()
        adapter_downloads = derivatives[derivatives['model_type'] == 'adapter']['download_count'].sum()
        lora_downloads = derivatives[derivatives['model_type'] == 'lora']['download_count'].sum()
        merge_downloads = derivatives[derivatives['model_type'] == 'merge']['download_count'].sum()
        other_downloads = derivatives[derivatives['model_type'] == 'other']['download_count'].sum()

        # 收集统计数据
        stats_row = {
            'base_model': base_model_id,
            'total_derivatives': len(derivatives),
            'quantized_count': type_counts.get('quantized', 0),
            'quantized_downloads': int(quantized_downloads),
            'finetune_count': type_counts.get('finetune', 0),
            'finetune_downloads': int(finetune_downloads),
            'adapter_count': type_counts.get('adapter', 0),
            'adapter_downloads': int(adapter_downloads),
            'lora_count': type_counts.get('lora', 0),
            'lora_downloads': int(lora_downloads),
            'merge_count': type_counts.get('merge', 0),
            'merge_downloads': int(merge_downloads),
            'other_count': type_counts.get('other', 0),
            'other_downloads': int(other_downloads),
            'total_derivative_downloads': int(total_deriv_downloads),
            'avg_downloads_per_derivative': int(avg_deriv_downloads),
            'top_derivative': max_deriv['model_id'],
            'top_derivative_downloads': int(max_deriv['download_count'])
        }
        stats_data.append(stats_row)

        print()

    # 4. 创建汇总统计表
    stats_df = pd.DataFrame(stats_data)

    print(f"{'='*80}")
    print(f"📊 汇总统计表")
    print(f"{'='*80}")
    print(stats_df.to_string(index=False))

    # 5. 全局类型分布
    print(f"\n{'='*80}")
    print(f"📈 全局衍生模型类型分布")
    print(f"{'='*80}")
    global_type_counts = derivative_models['model_type'].value_counts()
    for model_type, count in global_type_counts.items():
        percentage = (count / len(derivative_models)) * 100
        print(f"  {model_type:15s}: {count:4d} ({percentage:5.1f}%)")

    # 6. 保存统计结果到新的 Excel
    output_file = excel_file.replace('.xlsx', '_统计.xlsx')
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: 汇总统计
        stats_df.to_excel(writer, sheet_name='汇总统计', index=False)

        # Sheet 2: 原始数据
        df.to_excel(writer, sheet_name='原始数据', index=False)

        # Sheet 3: 衍生模型详细列表（按 base model 排序）
        derivative_models_sorted = derivative_models.sort_values(['base_model', 'download_count'], ascending=[True, False])
        cols_to_export = ['model_id', 'base_model', 'model_type', 'model_category', 'publisher', 'download_count', 'likes', 'created_at']
        derivative_models_sorted[cols_to_export].to_excel(writer, sheet_name='衍生模型列表', index=False)

        # Sheet 4-N: 每个 base model 单独一个 sheet
        for base_model_id in base_models['model_id'].unique():
            derivatives = df[(df['base_model'] == base_model_id) & (df['is_base'] == False)]
            if len(derivatives) > 0:
                sheet_name = base_model_id.split('/')[-1][:31]  # Excel sheet 名称限制
                derivatives_sorted = derivatives.sort_values('download_count', ascending=False)
                derivatives_sorted[cols_to_export].to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n✅ 统计结果已保存到: {output_file}")
    print(f"   包含内容:")
    print(f"   - 汇总统计: 每个 base model 的类型统计")
    print(f"   - 原始数据: 完整数据")
    print(f"   - 衍生模型列表: 所有衍生模型的关键信息")
    print(f"   - 各 base model 独立 sheet: {len(base_models)} 个")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        excel_file = "qwen_model_tree_20251203_105128.xlsx"

    analyze_model_tree(excel_file)
