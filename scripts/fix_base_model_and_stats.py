"""
修复 Excel 文件中的 base_model 和 model_group，并重新生成统计汇总表
"""
import pandas as pd
import re
import sys
from typing import List


def extract_model_group(model_id: str) -> str:
    """
    提取模型分组名称：找到最后一次出现的"数字+B"，之后的内容去掉
    """
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
    pattern = r'[A]?\d+(?:\.\d+)?B'
    matches = list(re.finditer(pattern, model_name, re.IGNORECASE))

    if not matches:
        return model_name

    last_match = matches[-1]
    end_pos = last_match.end()
    group_name = model_name[:end_pos]

    return group_name


def infer_base_model(model_id: str, official_model_ids: List[str]) -> str:
    """
    从模型名称推断 base_model

    Args:
        model_id: 模型 ID
        official_model_ids: 官方模型 ID 列表

    Returns:
        str: 推断的 base_model，如果无法推断则返回空字符串
    """
    model_name_lower = model_id.lower()

    # 按官方模型名称长度降序排序，优先匹配更具体的模型
    sorted_official = sorted(official_model_ids,
                            key=lambda x: len(x.split('/')[-1]),
                            reverse=True)

    for official_id in sorted_official:
        official_name = official_id.split('/')[-1].lower()

        # 移除常见后缀进行匹配
        official_core = official_name.replace('-pt', '').replace('-paddle', '').replace('-base', '')

        if official_core in model_name_lower:
            return official_id

    return ''


def fix_excel_file(excel_file: str):
    """修复 Excel 文件"""
    print(f"📂 读取文件: {excel_file}")

    xls = pd.ExcelFile(excel_file)
    sheet_names = xls.sheet_names

    print(f"✅ 找到 {len(sheet_names)} 个 sheet: {sheet_names}")

    # 读取所有 sheet
    sheets = {}
    for sheet_name in sheet_names:
        sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)

    # 收集所有官方模型 ID
    all_official_ids = []
    for sheet_name in sheet_names:
        if sheet_name == '统计汇总':
            continue
        df = sheets[sheet_name]
        official_ids = df[df['is_base'] == True]['model_id'].tolist()
        all_official_ids.extend(official_ids)

    print(f"\n找到 {len(all_official_ids)} 个官方模型")

    # 修复每个数据 sheet
    for sheet_name in sheet_names:
        if sheet_name == '统计汇总':
            continue

        print(f"\n{'='*80}")
        print(f"🔧 处理 sheet: {sheet_name}")
        print(f"{'='*80}")

        df = sheets[sheet_name]

        # 获取该 sheet 的官方模型
        sheet_official_ids = df[df['is_base'] == True]['model_id'].tolist()

        # 修复 base_model 和 model_group
        fixed_count = 0
        for idx, row in df.iterrows():
            if row['is_base']:
                continue

            # 如果 base_model 是空的，尝试推断
            if pd.isna(row['base_model']) or row['base_model'] == '':
                inferred_base = infer_base_model(row['model_id'], sheet_official_ids)
                if inferred_base:
                    df.at[idx, 'base_model'] = inferred_base
                    df.at[idx, 'model_group'] = extract_model_group(inferred_base)
                    fixed_count += 1
                    print(f"  ✅ {row['model_id']}")
                    print(f"     推断 base_model: {inferred_base}")
                else:
                    # 无法推断，使用模型自己的 group
                    df.at[idx, 'model_group'] = extract_model_group(row['model_id'])
                    print(f"  ⚠️  {row['model_id']}")
                    print(f"     无法推断 base_model，使用自身提取的 group")

        print(f"\n修复了 {fixed_count} 个模型的 base_model")

        sheets[sheet_name] = df

    # 重新生成统计汇总表
    print(f"\n{'='*80}")
    print("📊 重新生成统计汇总表")
    print(f"{'='*80}")

    def create_stats(df, series_name):
        """创建统计汇总表"""
        stats_data = []

        # 按 model_group 分组统计
        for group_name in df[df['is_base'] == True]['model_group'].unique():
            # 获取该分组的所有衍生模型（包括推断得到的）
            derivatives = df[(df['model_group'] == group_name) & (df['is_base'] == False)]

            if len(derivatives) == 0:
                continue

            # 统计各类型数量和下载量
            type_stats = {}
            for model_type in ['quantized', 'finetune', 'adapter', 'lora', 'merge', 'other']:
                type_models = derivatives[derivatives['model_type'] == model_type]
                type_stats[f'{model_type}_count'] = len(type_models)
                type_stats[f'{model_type}_downloads'] = int(type_models['download_count'].sum())

            # 总计
            total_derivatives = len(derivatives)
            total_downloads = int(derivatives['download_count'].sum())

            # 计算百分比
            for model_type in ['quantized', 'finetune', 'adapter', 'lora', 'merge', 'other']:
                count = type_stats[f'{model_type}_count']
                downloads = type_stats[f'{model_type}_downloads']
                type_stats[f'{model_type}_count_pct'] = f"{count/total_derivatives*100:.1f}%" if total_derivatives > 0 else "0%"
                type_stats[f'{model_type}_downloads_pct'] = f"{downloads/total_downloads*100:.1f}%" if total_downloads > 0 else "0%"

            stats_row = {
                'series': series_name,
                'model_group': group_name,
                'total_derivatives': total_derivatives,
                'total_downloads': total_downloads,
                **type_stats
            }
            stats_data.append(stats_row)

            print(f"  {group_name}: {total_derivatives} 个衍生模型, {total_downloads:,} 总下载量")

        return pd.DataFrame(stats_data)

    # 生成各个系列的统计
    all_stats = []
    for sheet_name in sheet_names:
        if sheet_name == '统计汇总':
            continue
        df = sheets[sheet_name]
        stats = create_stats(df, sheet_name)
        all_stats.append(stats)

    stats_combined = pd.concat(all_stats, ignore_index=True)
    sheets['统计汇总'] = stats_combined

    # 保存修复后的文件
    print(f"\n{'='*80}")
    print("💾 保存修复后的文件")
    print(f"{'='*80}")

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✅ {sheet_name}: {len(df)} 行")

    print(f"\n✅ 文件已更新: {excel_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        print("用法: python fix_base_model_and_stats.py <excel_file>")
        sys.exit(1)

    fix_excel_file(excel_file)
