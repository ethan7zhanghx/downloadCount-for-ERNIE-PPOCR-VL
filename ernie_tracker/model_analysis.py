"""
模型分析模块 - 推断 base_model 和统计衍生生态
"""
import re
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# 官方模型分组配置
OFFICIAL_MODEL_GROUPS = {
    'ERNIE-4.5-0.3B': {
        'models': [
            'baidu/ERNIE-4.5-0.3B-PT',
            'baidu/ERNIE-4.5-0.3B-Base-PT',
            'baidu/ERNIE-4.5-0.3B-Paddle',
            'baidu/ERNIE-4.5-0.3B-Base-Paddle',
        ],
        'pattern': r'ERNIE-4\.5-0\.3B(?!.*Thinking)',
    },
    'ERNIE-4.5-21B-A3B': {
        'models': [
            'baidu/ERNIE-4.5-21B-A3B-PT',
            'baidu/ERNIE-4.5-21B-A3B-Base-PT',
            'baidu/ERNIE-4.5-21B-A3B-Paddle',
            'baidu/ERNIE-4.5-21B-A3B-Base-Paddle',
        ],
        'pattern': r'ERNIE-4\.5-21B-A3B(?!.*Thinking)',
    },
    'ERNIE-4.5-21B-A3B-Thinking': {
        'models': [
            'baidu/ERNIE-4.5-21B-A3B-Thinking',
        ],
        'pattern': r'ERNIE-4\.5-21B-A3B-Thinking',
    },
    'ERNIE-4.5-VL-28B-A3B': {
        'models': [
            'baidu/ERNIE-4.5-VL-28B-A3B-PT',
            'baidu/ERNIE-4.5-VL-28B-A3B-Base-PT',
            'baidu/ERNIE-4.5-VL-28B-A3B-Paddle',
            'baidu/ERNIE-4.5-VL-28B-A3B-Base-Paddle',
        ],
        'pattern': r'ERNIE-4\.5-VL-28B-A3B(?!.*Thinking)',
    },
    'ERNIE-4.5-VL-28B-A3B-Thinking': {
        'models': [
            'baidu/ERNIE-4.5-VL-28B-A3B-Thinking',
        ],
        'pattern': r'ERNIE-4\.5-VL-28B-A3B-Thinking',
    },
    'ERNIE-4.5-300B-A47B': {
        'models': [
            'baidu/ERNIE-4.5-300B-A47B-PT',
            'baidu/ERNIE-4.5-300B-A47B-Base-PT',
            'baidu/ERNIE-4.5-300B-A47B-Paddle',
            'baidu/ERNIE-4.5-300B-A47B-Base-Paddle',
            'baidu/ERNIE-4.5-300B-A47B-FP8-Paddle',
            'baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle',
            'baidu/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle',
        ],
        'pattern': r'ERNIE-4\.5-300B-A47B',
    },
    'ERNIE-4.5-VL-424B-A47B': {
        'models': [
            'baidu/ERNIE-4.5-VL-424B-A47B-PT',
            'baidu/ERNIE-4.5-VL-424B-A47B-Base-PT',
            'baidu/ERNIE-4.5-VL-424B-A47B-Paddle',
            'baidu/ERNIE-4.5-VL-424B-A47B-Base-Paddle',
        ],
        'pattern': r'ERNIE-4\.5-VL-424B-A47B',
    },
    'PaddleOCR-VL': {
        'models': [
            'PaddlePaddle/PaddleOCR-VL',
        ],
        'pattern': r'PaddleOCR-VL',
    },
}


# 构建官方 base_model 的大小写无关映射，便于标准化
CANONICAL_BASE_MODEL_MAP = {}
for group_info in OFFICIAL_MODEL_GROUPS.values():
    for model_id in group_info['models']:
        CANONICAL_BASE_MODEL_MAP[model_id.lower()] = model_id
        CANONICAL_BASE_MODEL_MAP[model_id.split('/')[-1].lower()] = model_id


def normalize_base_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    标准化 base_model，修复 PaddleOCR-VL 错归到 ERNIE 的问题

    - 统一大小写、补全 publisher 前缀
    - 对官方原始模型清空 base_model，避免被当成衍生
    - 将 PaddleOCR-VL 相关模型的 base_model 归一到官方 ID
    """
    normalized_df = df.copy()
    stats = {'canonicalized': 0, 'cleared_original': 0, 'paddleocr_fixed': 0}

    if 'base_model' not in normalized_df.columns:
        normalized_df['base_model'] = None
        return normalized_df, stats

    def clean_value(val):
        if pd.isna(val):
            return None
        val_str = str(val).strip()
        return None if val_str.lower() in ['', 'none', 'nan'] else val_str

    normalized_df['base_model'] = normalized_df['base_model'].apply(clean_value)

    # 官方原始模型不应带 base_model，避免计入衍生
    if 'model_type' in normalized_df.columns:
        cleared_mask = (normalized_df['model_type'] == 'original') & normalized_df['base_model'].notna()
        stats['cleared_original'] += int(cleared_mask.sum())
        normalized_df.loc[normalized_df['model_type'] == 'original', 'base_model'] = None

    if 'data_source' in normalized_df.columns:
        source_mask = (normalized_df['data_source'] == 'original') & normalized_df['base_model'].notna()
        stats['cleared_original'] += int(source_mask.sum())
        normalized_df.loc[normalized_df['data_source'] == 'original', 'base_model'] = None

    def canonicalize(val):
        if not val:
            return None
        lower = val.lower()
        canonical = CANONICAL_BASE_MODEL_MAP.get(lower)
        if not canonical:
            bare = lower.split('/')[-1]
            canonical = CANONICAL_BASE_MODEL_MAP.get(bare)
        if canonical and canonical != val:
            stats['canonicalized'] += 1
            return canonical
        return val

    normalized_df['base_model'] = normalized_df['base_model'].apply(canonicalize)

    # 专门修复 PaddleOCR-VL 被误判成 ERNIE-4.5 的情况
    paddle_base = OFFICIAL_MODEL_GROUPS['PaddleOCR-VL']['models'][0]

    def is_paddleocr(row):
        name = str(row.get('model_name', '')).lower()
        category = str(row.get('model_category', '')).lower()
        base = str(row.get('base_model') or '').lower()
        publisher = str(row.get('publisher', '')).lower()
        return (
            'paddleocr-vl' in name
            or 'paddleocr-vl' in category
            or 'paddleocr-vl' in base
            or publisher == 'paddleocr-vl'
        )

    paddle_mask = normalized_df.apply(is_paddleocr, axis=1)
    base_col = normalized_df['base_model'].fillna('')
    wrong_base_mask = base_col.str.contains('ernie-4.5', case=False, na=False) | base_col.str.fullmatch('paddleocr-vl', case=False, na=False)
    fix_mask = paddle_mask & (normalized_df['base_model'].isna() | wrong_base_mask)
    if 'model_type' in normalized_df.columns:
        fix_mask = fix_mask & (normalized_df['model_type'] != 'original')
    if 'data_source' in normalized_df.columns:
        fix_mask = fix_mask & (normalized_df['data_source'] != 'original')
    stats['paddleocr_fixed'] = int(fix_mask.sum())
    normalized_df.loc[fix_mask, 'base_model'] = paddle_base

    return normalized_df, stats


def infer_base_model_from_name(model_name: str, publisher: str, full_model_id: str = None) -> Optional[str]:
    """
    根据模型名称推断 base_model

    Args:
        model_name: 模型名称（不含publisher）
        publisher: 发布者
        full_model_id: 完整模型ID（publisher/model_name），可选

    Returns:
        推断出的 base_model，如果无法推断则返回 None
    """
    if full_model_id is None:
        full_model_id = f"{publisher}/{model_name}"

    # 如果是官方模型，不需要推断
    if publisher.lower() in ['baidu', 'paddlepaddle']:
        return None

    # 将模型名称转为小写便于匹配
    model_lower = full_model_id.lower()

    # 按优先级匹配（从具体到一般）
    # 1. 先匹配 Thinking 模型（优先级最高）
    if 'thinking' in model_lower:
        if '21b' in model_lower or '21-b' in model_lower:
            return 'baidu/ERNIE-4.5-21B-A3B-Thinking'
        elif 'vl' in model_lower and ('28b' in model_lower or '28-b' in model_lower):
            return 'baidu/ERNIE-4.5-VL-28B-A3B-Thinking'

    # 2. 匹配其他尺寸模型
    # VL-424B-A47B
    if 'vl' in model_lower and ('424b' in model_lower or '424-b' in model_lower):
        # 优先匹配 PT 版本
        if 'paddle' in model_lower:
            return 'baidu/ERNIE-4.5-VL-424B-A47B-Paddle'
        else:
            return 'baidu/ERNIE-4.5-VL-424B-A47B-PT'

    # 300B-A47B
    if '300b' in model_lower or '300-b' in model_lower:
        if 'paddle' in model_lower:
            return 'baidu/ERNIE-4.5-300B-A47B-Paddle'
        else:
            return 'baidu/ERNIE-4.5-300B-A47B-PT'

    # VL-28B-A3B (非Thinking)
    if 'vl' in model_lower and ('28b' in model_lower or '28-b' in model_lower):
        if 'paddle' in model_lower:
            return 'baidu/ERNIE-4.5-VL-28B-A3B-Paddle'
        else:
            return 'baidu/ERNIE-4.5-VL-28B-A3B-PT'

    # 21B-A3B (非Thinking)
    if '21b' in model_lower or '21-b' in model_lower:
        if 'paddle' in model_lower:
            return 'baidu/ERNIE-4.5-21B-A3B-Paddle'
        else:
            return 'baidu/ERNIE-4.5-21B-A3B-PT'

    # 0.3B
    if '0.3b' in model_lower or '0-3b' in model_lower or '300m' in model_lower:
        if 'paddle' in model_lower:
            return 'baidu/ERNIE-4.5-0.3B-Paddle'
        else:
            return 'baidu/ERNIE-4.5-0.3B-PT'

    # PaddleOCR-VL
    if 'paddleocr' in model_lower and 'vl' in model_lower:
        return 'PaddlePaddle/PaddleOCR-VL'

    # 无法推断
    return None


def get_model_group(base_model: str) -> Optional[str]:
    """
    根据 base_model 确定它属于哪个分组

    Args:
        base_model: 基础模型ID（如 'baidu/ERNIE-4.5-21B-A3B-PT'）

    Returns:
        分组名称，如果不属于任何分组则返回 None
    """
    if not base_model:
        return None

    for group_name, group_info in OFFICIAL_MODEL_GROUPS.items():
        if base_model in group_info['models']:
            return group_name

    return None


def analyze_derivative_ecosystem(df: pd.DataFrame, infer_missing: bool = True) -> Dict:
    """
    分析衍生模型生态

    Args:
        df: 包含模型数据的 DataFrame（必须包含 base_model, model_type 列）
        infer_missing: 是否推断缺失的 base_model

    Returns:
        分析结果字典
    """
    # 复制数据避免修改原始数据
    analysis_df, normalization_stats = normalize_base_models(df)

    if any(normalization_stats.values()):
        print(
            f"🔧 标准化 base_model | "
            f"清理官方: {normalization_stats['cleared_original']} | "
            f"ID归一: {normalization_stats['canonicalized']} | "
            f"PaddleOCR修正: {normalization_stats['paddleocr_fixed']}"
        )

    # 1. 推断缺失的 base_model
    if infer_missing:
        print("🔍 推断缺失的 base_model...")
        inferred_count = 0

        for idx, row in analysis_df.iterrows():
            # 只处理没有 base_model 的记录
            if pd.isna(row.get('base_model')) or not row.get('base_model'):
                model_name = row.get('model_name', '')
                publisher = row.get('publisher', '')

                # 推断 base_model
                inferred_base = infer_base_model_from_name(model_name, publisher)

                if inferred_base:
                    analysis_df.at[idx, 'base_model'] = inferred_base
                    analysis_df.at[idx, 'base_model_inferred'] = True
                    inferred_count += 1

        print(f"  ✅ 成功推断 {inferred_count} 个模型的 base_model")

    # 2. 按分组统计
    print("\n📊 按分组统计衍生生态...")

    # 过滤出有 base_model 的记录（衍生模型）
    derivatives = analysis_df[
        analysis_df['base_model'].notna() &
        (analysis_df['base_model'] != '') &
        (analysis_df['base_model'] != 'None')
    ].copy()

    print(f"  ✅ 共有 {len(derivatives)} 个衍生模型")

    # 添加分组信息
    derivatives['model_group'] = derivatives['base_model'].apply(get_model_group)

    # 统计结果
    results = {}

    for group_name in OFFICIAL_MODEL_GROUPS.keys():
        group_derivatives = derivatives[derivatives['model_group'] == group_name]

        if len(group_derivatives) == 0:
            results[group_name] = {
                'total': 0,
                'by_type': {},
                'by_data_source': {},
                'models': []
            }
            continue

        # 按类型统计
        type_counts = group_derivatives['model_type'].value_counts().to_dict()

        # 按数据来源统计（如果有 data_source 列）
        source_counts = {}
        if 'data_source' in group_derivatives.columns:
            source_counts = group_derivatives['data_source'].value_counts().to_dict()

        # 获取样本模型
        sample_models = group_derivatives[['model_name', 'publisher', 'base_model', 'model_type', 'download_count']].head(10).to_dict('records')

        results[group_name] = {
            'total': len(group_derivatives),
            'by_type': type_counts,
            'by_data_source': source_counts,
            'models': sample_models,
            'base_models': group_derivatives['base_model'].unique().tolist()
        }

    # 3. 总体统计
    summary = {
        'total_derivatives': len(derivatives),
        'total_inferred': inferred_count if infer_missing else 0,
        'by_group': results,
        'overall_by_type': derivatives['model_type'].value_counts().to_dict(),
    }

    return summary


def print_analysis_report(analysis_result: Dict):
    """
    打印分析报告

    Args:
        analysis_result: analyze_derivative_ecosystem() 的返回结果
    """
    print("\n" + "="*80)
    print("📊 ERNIE-4.5 衍生模型生态分析报告")
    print("="*80)

    print(f"\n📈 总体统计:")
    print(f"  - 衍生模型总数: {analysis_result['total_derivatives']}")
    print(f"  - 推断的 base_model: {analysis_result['total_inferred']}")

    print(f"\n📊 整体类型分布:")
    for model_type, count in sorted(analysis_result['overall_by_type'].items(), key=lambda x: x[1], reverse=True):
        emoji = {
            'quantized': '⚡',
            'finetune': '🔧',
            'adapter': '🔌',
            'lora': '🎯',
            'merge': '🔀',
            'other': '📦'
        }.get(model_type, '📦')
        print(f"  {emoji} {model_type}: {count} 个")

    print(f"\n" + "="*80)
    print("📋 各分组详细统计")
    print("="*80)

    for group_name, group_data in analysis_result['by_group'].items():
        print(f"\n🏷️  {group_name}")
        print(f"  总计: {group_data['total']} 个衍生模型")

        if group_data['total'] > 0:
            print(f"\n  按类型分布:")
            for model_type, count in sorted(group_data['by_type'].items(), key=lambda x: x[1], reverse=True):
                emoji = {
                    'quantized': '⚡',
                    'finetune': '🔧',
                    'adapter': '🔌',
                    'lora': '🎯',
                    'merge': '🔀',
                    'other': '📦'
                }.get(model_type, '📦')
                percentage = (count / group_data['total']) * 100
                print(f"    {emoji} {model_type}: {count} 个 ({percentage:.1f}%)")

            if group_data['by_data_source']:
                print(f"\n  按数据来源分布:")
                for source, count in group_data['by_data_source'].items():
                    source_label = {
                        'search': '搜索发现',
                        'model_tree': 'Model Tree',
                        'both': '搜索+Model Tree',
                        None: '推断'
                    }.get(source, source)
                    print(f"    - {source_label}: {count} 个")

            print(f"\n  包含的官方 base_model:")
            for base_model in group_data['base_models']:
                print(f"    - {base_model}")

            print(f"\n  样本模型（前5个）:")
            for i, model in enumerate(group_data['models'][:5], 1):
                print(f"    {i}. {model['publisher']}/{model['model_name']}")
                downloads = model.get('download_count', 0)
                downloads_int = int(downloads) if pd.notna(downloads) else 0
                print(f"       类型: {model['model_type']} | base: {model['base_model']} | 下载: {downloads_int:,}")

        print()


def export_analysis_to_excel(analysis_result: Dict, df: pd.DataFrame, output_path: str):
    """
    导出分析结果到 Excel（多个sheet）

    Args:
        analysis_result: 分析结果
        df: 原始数据
        output_path: 输出文件路径
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: 总体统计
        summary_data = {
            '指标': ['衍生模型总数', '推断的base_model数量'],
            '数值': [analysis_result['total_derivatives'], analysis_result['total_inferred']]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='总体统计', index=False)

        # Sheet 2: 各分组统计
        group_stats = []
        for group_name, group_data in analysis_result['by_group'].items():
            for model_type, count in group_data['by_type'].items():
                group_stats.append({
                    '分组': group_name,
                    '模型类型': model_type,
                    '数量': count,
                    '占比': f"{(count / group_data['total'] * 100):.1f}%" if group_data['total'] > 0 else "0%"
                })

        if group_stats:
            pd.DataFrame(group_stats).to_excel(writer, sheet_name='分组统计', index=False)

        # Sheet 3-N: 每个分组的详细模型列表
        derivatives = df[
            df['base_model'].notna() &
            (df['base_model'] != '') &
            (df['base_model'] != 'None')
        ].copy()

        derivatives['model_group'] = derivatives['base_model'].apply(get_model_group)

        for group_name in OFFICIAL_MODEL_GROUPS.keys():
            group_derivatives = derivatives[derivatives['model_group'] == group_name]

            if len(group_derivatives) > 0:
                # 选择重要列
                export_cols = ['model_name', 'publisher', 'base_model', 'model_type',
                             'download_count', 'data_source', 'model_category']
                available_cols = [col for col in export_cols if col in group_derivatives.columns]

                sheet_df = group_derivatives[available_cols].sort_values('download_count', ascending=False)

                # Excel sheet 名称长度限制为31
                sheet_name = group_name[:28] + '...' if len(group_name) > 31 else group_name
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n✅ 分析结果已导出到: {output_path}")


if __name__ == "__main__":
    # 测试推断功能
    print("🧪 测试 base_model 推断功能:\n")

    test_cases = [
        ("baidu_ERNIE-4.5-21B-A3B-PT-GGUF", "bartowski"),
        ("ERNIE-4.5-21B-A3B-Thinking-GGUF", "unsloth"),
        ("ERNIE-4.5-0.3B-PT-GGUF", "lmstudio-community"),
        ("ERNIE-4.5-VL-28B-A3B-Thinking-GGUF", "gabriellarson"),
        ("ERNIE-4.5-300B-A47B-PT-GGUF", "unsloth"),
        ("some-random-model", "user123"),
    ]

    for model_name, publisher in test_cases:
        inferred = infer_base_model_from_name(model_name, publisher)
        print(f"  {publisher}/{model_name}")
        print(f"    → {inferred if inferred else '无法推断'}\n")
