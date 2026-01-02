"""
获取 Qwen 模型的 Model Tree 数据
这是一个独立的脚本，不集成到下载数据统计系统中
"""
from huggingface_hub import list_models, model_info
from datetime import datetime
import pandas as pd
import json
from typing import List, Dict
from ernie_tracker.fetchers.fetchers_modeltree import classify_model, classify_model_type


# 要获取的 Qwen 模型列表
# Qwen3 系列
QWEN3_MODELS = [
    # 2507 系列
    "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507-FP8",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-4B-Instruct-2507-FP8",
    "Qwen/Qwen3-4B-Instruct-2507",
    # 基础系列
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-0.6B",
    # FP8 系列
    "Qwen/Qwen3-235B-A22B-FP8",
    "Qwen/Qwen3-30B-A3B-FP8",
    "Qwen/Qwen3-32B-FP8",
    "Qwen/Qwen3-14B-FP8",
    "Qwen/Qwen3-8B-FP8",
    "Qwen/Qwen3-4B-FP8",
    "Qwen/Qwen3-1.7B-FP8",
    "Qwen/Qwen3-0.6B-FP8",
    # GPTQ/AWQ 系列
    "Qwen/Qwen3-235B-A22B-GPTQ-Int4",
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
    "Qwen/Qwen3-32B-AWQ",
    "Qwen/Qwen3-14B-AWQ",
    "Qwen/Qwen3-8B-AWQ",
    "Qwen/Qwen3-4B-AWQ",
    "Qwen/Qwen3-1.7B-GPTQ-Int8",
    "Qwen/Qwen3-0.6B-GPTQ-Int8",
    # GGUF 系列
    "Qwen/Qwen3-235B-A22B-GGUF",
    "Qwen/Qwen3-30B-A3B-GGUF",
    "Qwen/Qwen3-32B-GGUF",
    "Qwen/Qwen3-14B-GGUF",
    "Qwen/Qwen3-8B-GGUF",
    "Qwen/Qwen3-4B-GGUF",
    "Qwen/Qwen3-1.7B-GGUF",
    "Qwen/Qwen3-0.6B-GGUF",
    # Base 系列
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-0.6B-Base",
    # MLX 系列
    "Qwen/Qwen3-4B-MLX-8bit",
    "Qwen/Qwen3-4B-MLX-bf16",
    "Qwen/Qwen3-4B-MLX-6bit",
    "Qwen/Qwen3-4B-MLX-4bit",
    "Qwen/Qwen3-8B-MLX-4bit",
    "Qwen/Qwen3-8B-MLX-6bit",
    "Qwen/Qwen3-8B-MLX-8bit",
    "Qwen/Qwen3-8B-MLX-bf16",
    "Qwen/Qwen3-0.6B-MLX-6bit",
    "Qwen/Qwen3-0.6B-MLX-4bit",
    "Qwen/Qwen3-0.6B-MLX-bf16",
    "Qwen/Qwen3-0.6B-MLX-8bit",
    "Qwen/Qwen3-32B-MLX-8bit",
    "Qwen/Qwen3-1.7B-MLX-6bit",
    "Qwen/Qwen3-1.7B-MLX-bf16",
    "Qwen/Qwen3-1.7B-MLX-8bit",
    "Qwen/Qwen3-1.7B-MLX-4bit",
    "Qwen/Qwen3-14B-MLX-6bit",
    "Qwen/Qwen3-14B-MLX-8bit",
    "Qwen/Qwen3-14B-MLX-4bit",
    "Qwen/Qwen3-14B-MLX-bf16",
    "Qwen/Qwen3-32B-MLX-6bit",
    "Qwen/Qwen3-32B-MLX-bf16",
    "Qwen/Qwen3-32B-MLX-4bit",
    "Qwen/Qwen3-30B-A3B-MLX-4bit",
    "Qwen/Qwen3-30B-A3B-MLX-bf16",
    "Qwen/Qwen3-30B-A3B-MLX-8bit",
    "Qwen/Qwen3-30B-A3B-MLX-6bit",
    "Qwen/Qwen3-235B-A22B-MLX-bf16",
    "Qwen/Qwen3-235B-A22B-MLX-6bit",
    "Qwen/Qwen3-235B-A22B-MLX-4bit",
    "Qwen/Qwen3-235B-A22B-MLX-8bit",
]

# Qwen3-VL 系列
QWEN3_VL_MODELS = [
    # Thinking/Instruct 系列
    "Qwen/Qwen3-VL-235B-A22B-Thinking",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
    "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking-FP8",
    "Qwen/Qwen3-VL-8B-Instruct-FP8",
    "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-4B-Thinking-FP8",
    "Qwen/Qwen3-VL-4B-Instruct-FP8",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-2B-Thinking",
    "Qwen/Qwen3-VL-2B-Thinking-FP8",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen/Qwen3-VL-32B-Thinking-FP8",
    "Qwen/Qwen3-VL-32B-Instruct-FP8",
    "Qwen/Qwen3-VL-2B-Instruct-FP8",
    # GGUF 系列
    "Qwen/Qwen3-VL-2B-Instruct-GGUF",
    "Qwen/Qwen3-VL-4B-Instruct-GGUF",
    "Qwen/Qwen3-VL-4B-Thinking-GGUF",
    "Qwen/Qwen3-VL-8B-Instruct-GGUF",
    "Qwen/Qwen3-VL-32B-Instruct-GGUF",
    "Qwen/Qwen3-VL-32B-Thinking-GGUF",
    "Qwen/Qwen3-VL-235B-A22B-Instruct-GGUF",
    "Qwen/Qwen3-VL-235B-A22B-Thinking-GGUF",
    "Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF",
    "Qwen/Qwen3-VL-30B-A3B-Thinking-GGUF",
    "Qwen/Qwen3-VL-2B-Thinking-GGUF",
    "Qwen/Qwen3-VL-8B-Thinking-GGUF",
]


def extract_model_group(model_id: str) -> str:
    """
    提取模型分组名称：找到最后一次出现的"数字+B"，之后的内容去掉

    例如：
    - Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 → Qwen3-235B-A22B
    - Qwen/Qwen3-4B-MLX-4bit → Qwen3-4B
    - Qwen/Qwen3-VL-30B-A3B-Instruct → Qwen3-VL-30B-A3B

    Args:
        model_id: 完整的模型 ID

    Returns:
        str: 分组名称
    """
    import re

    # 去掉 "Qwen/" 前缀
    model_name = model_id.replace('Qwen/', '')

    # 匹配所有的"数字+B"模式（包括小数和 A22B/A3B 这种格式）
    # 匹配模式：可选的 A + 数字（可能包含小数点） + B
    # 例如：235B, 30B, A22B, A3B, 1.7B, 0.6B
    pattern = r'[A]?\d+(?:\.\d+)?B'

    # 找到所有匹配
    matches = list(re.finditer(pattern, model_name, re.IGNORECASE))

    if not matches:
        # 没有找到匹配，返回原始名称
        return model_name

    # 获取最后一个匹配
    last_match = matches[-1]
    end_pos = last_match.end()

    # 截取到最后一个"数字+B"的位置
    group_name = model_name[:end_pos]

    return group_name


def get_all_model_info_fields(model_id: str) -> Dict:
    """
    获取模型的所有信息字段（尽可能保留 API 返回的所有字段）

    Args:
        model_id: 模型ID

    Returns:
        Dict: 包含所有可用字段的字典
    """
    try:
        # 获取模型详细信息，使用 expand 参数获取更多字段
        info = model_info(model_id, expand=["downloadsAllTime", "trendingScore"])
        card_data = None
        if hasattr(info, 'cardData') and info.cardData:
            if isinstance(info.cardData, dict):
                card_data = info.cardData
            elif hasattr(info.cardData, '__dict__'):
                card_data = info.cardData.__dict__

        # 从 list_models 获取完整模型对象（包含 tags 等字段）
        model_obj = None
        try:
            models = list(list_models(model_name=model_id, full=True, limit=1))
            if models:
                model_obj = models[0]
        except Exception as e:
            print(f"    ⚠️ 无法从 list_models 获取 {model_id}: {e}")

        # 收集所有字段
        model_data = {}

        # 从 model_info 获取的字段
        info_fields = [
            'modelId', 'sha', 'author', 'private', 'disabled', 'gated',
            'downloads', 'downloads_all_time', 'likes', 'library_name',
            'pipeline_tag', 'created_at', 'last_modified', 'card_data',
            'siblings', 'spaces', 'safetensors', 'config'
        ]

        for field in info_fields:
            if hasattr(info, field):
                value = getattr(info, field)
                # 处理特殊类型
                if field in ['created_at', 'last_modified'] and value:
                    model_data[field] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                elif field == 'card_data' and value:
                    # 将 card_data 转换为字典
                    model_data[field] = value.__dict__ if hasattr(value, '__dict__') else str(value)
                elif field == 'siblings' and value:
                    # siblings 是文件列表
                    model_data[field] = [s.__dict__ if hasattr(s, '__dict__') else str(s) for s in value]
                elif field == 'config' and value:
                    # config 可能是字典或对象
                    model_data[field] = value if isinstance(value, dict) else (value.__dict__ if hasattr(value, '__dict__') else str(value))
                else:
                    model_data[field] = value

        # 从 model_obj 获取的字段（如果可用）
        if model_obj:
            model_obj_fields = ['tags', 'trending_score', 'sdk']
            for field in model_obj_fields:
                if hasattr(model_obj, field):
                    value = getattr(model_obj, field)
                    model_data[field] = value
        # 优先使用 model_info 提供的 tags，其次 fallback 到 model_obj 的 tags
        if hasattr(info, 'tags') and info.tags:
            model_data['tags'] = info.tags
        elif 'tags' not in model_data and hasattr(model_obj, 'tags'):
            model_data['tags'] = getattr(model_obj, 'tags')

        # 显式保存模型卡内容，方便后续使用
        if card_data and 'card_data' not in model_data:
            model_data['card_data'] = card_data

        # 添加我们系统使用的分类字段
        publisher = model_data.get('author', 'Unknown')
        tags = model_data.get('tags', [])
        pipeline_tag = model_data.get('pipeline_tag', None)

        model_data['model_category'] = classify_model(model_id, publisher)
        model_data['model_type'] = classify_model_type(model_id, tags, pipeline_tag, card_data)

        # 添加下载量统一字段（优先使用 downloads_all_time）
        model_data['download_count'] = model_data.get('downloads_all_time') or model_data.get('downloads', 0) or 0

        # 添加获取时间
        model_data['fetched_at'] = datetime.now().isoformat()

        return model_data

    except Exception as e:
        print(f"    ❌ 获取模型 {model_id} 信息失败: {e}")
        return None


def get_model_tree_with_full_info(base_model_id: str) -> Dict:
    """
    获取指定模型的完整 Model Tree 信息（包含所有 API 字段）

    Args:
        base_model_id: 基础模型ID

    Returns:
        Dict: 包含基础模型和所有衍生模型的完整信息
    """
    print(f"\n{'='*80}")
    print(f"📊 获取 {base_model_id} 的 Model Tree")
    print(f"{'='*80}")

    result = {
        'base_model_id': base_model_id,
        'base_model_info': None,
        'derivatives': [],
        'summary': {
            'total_derivatives': 0,
            'by_type': {},
            'by_category': {},
            'total_downloads': 0
        }
    }

    # 1. 获取基础模型的完整信息
    print(f"\n1️⃣ 获取基础模型信息...")
    base_info = get_all_model_info_fields(base_model_id)
    if not base_info:
        print(f"  ❌ 无法获取基础模型 {base_model_id} 的信息")
        return result

    result['base_model_info'] = base_info
    print(f"  ✅ 基础模型信息获取成功")
    print(f"     下载量: {base_info.get('download_count', 0):,}")
    print(f"     分类: {base_info.get('model_category', 'N/A')}")
    print(f"     类型: {base_info.get('model_type', 'N/A')}")

    # 2. 获取衍生模型列表
    print(f"\n2️⃣ 查找衍生模型...")
    try:
        derivatives = list(list_models(
            filter=f"base_model:{base_model_id}",
            full=True,
            limit=1000
        ))

        if not derivatives:
            print(f"  ⚪ 没有找到衍生模型")
            return result

        print(f"  ✅ 找到 {len(derivatives)} 个衍生模型")

    except Exception as e:
        print(f"  ❌ 查找衍生模型失败: {e}")
        return result

    # 3. 获取每个衍生模型的完整信息
    print(f"\n3️⃣ 获取衍生模型详细信息...")
    for idx, deriv in enumerate(derivatives, 1):
        print(f"  [{idx}/{len(derivatives)}] {deriv.id}")

        deriv_info = get_all_model_info_fields(deriv.id)
        if deriv_info:
            # 确保 model_id 被正确设置（使用 deriv.id）
            if 'modelId' not in deriv_info or not deriv_info['modelId']:
                deriv_info['modelId'] = deriv.id
            # 确保 author 被正确设置
            if 'author' not in deriv_info or not deriv_info['author']:
                deriv_info['author'] = deriv.author if hasattr(deriv, 'author') else ''

            deriv_info['base_model'] = base_model_id
            result['derivatives'].append(deriv_info)

            # 更新统计
            model_type = deriv_info.get('model_type', 'other')
            model_category = deriv_info.get('model_category', 'other')
            downloads = deriv_info.get('download_count', 0)

            result['summary']['by_type'][model_type] = result['summary']['by_type'].get(model_type, 0) + 1
            result['summary']['by_category'][model_category] = result['summary']['by_category'].get(model_category, 0) + 1
            result['summary']['total_downloads'] += downloads

            print(f"     ✅ 下载量: {downloads:,} | 类型: {model_type} | 分类: {model_category}")

    result['summary']['total_derivatives'] = len(result['derivatives'])

    # 4. 打印汇总统计
    print(f"\n{'='*80}")
    print(f"📊 {base_model_id} - Model Tree 汇总")
    print(f"{'='*80}")
    print(f"衍生模型总数: {result['summary']['total_derivatives']}")
    print(f"总下载量: {result['summary']['total_downloads']:,}")
    print(f"\n按类型分布:")
    for model_type, count in sorted(result['summary']['by_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_type}: {count}")
    print(f"\n按分类分布:")
    for category, count in sorted(result['summary']['by_category'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")

    return result


def main():
    """
    主函数：获取所有 Qwen 模型的 Model Tree 数据
    """
    print("🚀 开始获取 Qwen 模型的 Model Tree 数据")
    print(f"目标模型数: {len(QWEN_MODELS)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    for model_id in QWEN_MODELS:
        result = get_model_tree_with_full_info(model_id)
        all_results[model_id] = result

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. 保存为 JSON（包含所有原始字段）
    json_filename = f"qwen_model_tree_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 JSON 数据已保存到: {json_filename}")

    # 2. 保存为 Excel（扁平化的表格数据）
    excel_data = []
    for base_model_id, result in all_results.items():
        # 添加基础模型行
        if result['base_model_info']:
            base_row = {
                'model_id': base_model_id,  # 第一列
                'base_model': base_model_id,
                'is_base': True,
                'model_name': base_model_id.split('/')[-1],
                'publisher': result['base_model_info'].get('author', ''),
                'model_type': result['base_model_info'].get('model_type', ''),
                'model_category': result['base_model_info'].get('model_category', ''),
                'download_count': result['base_model_info'].get('download_count', 0),
                'downloads_all_time': result['base_model_info'].get('downloads_all_time', 0),
                'downloads': result['base_model_info'].get('downloads', 0),
                'likes': result['base_model_info'].get('likes', 0),
                'pipeline_tag': result['base_model_info'].get('pipeline_tag', ''),
                'library_name': result['base_model_info'].get('library_name', ''),
                'tags': str(result['base_model_info'].get('tags', [])),
                'created_at': result['base_model_info'].get('created_at', ''),
                'last_modified': result['base_model_info'].get('last_modified', ''),
                'trending_score': result['base_model_info'].get('trending_score', 0),
                'fetched_at': result['base_model_info'].get('fetched_at', '')
            }
            excel_data.append(base_row)

        # 添加衍生模型行
        for deriv_info in result['derivatives']:
            deriv_row = {
                'model_id': deriv_info.get('modelId', ''),  # 第一列
                'base_model': base_model_id,
                'is_base': False,
                'model_name': deriv_info.get('modelId', '').split('/')[-1] if deriv_info.get('modelId') else '',
                'publisher': deriv_info.get('author', ''),
                'model_type': deriv_info.get('model_type', ''),
                'model_category': deriv_info.get('model_category', ''),
                'download_count': deriv_info.get('download_count', 0),
                'downloads_all_time': deriv_info.get('downloads_all_time', 0),
                'downloads': deriv_info.get('downloads', 0),
                'likes': deriv_info.get('likes', 0),
                'pipeline_tag': deriv_info.get('pipeline_tag', ''),
                'library_name': deriv_info.get('library_name', ''),
                'tags': str(deriv_info.get('tags', [])),
                'created_at': deriv_info.get('created_at', ''),
                'last_modified': deriv_info.get('last_modified', ''),
                'trending_score': deriv_info.get('trending_score', 0),
                'fetched_at': deriv_info.get('fetched_at', '')
            }
            excel_data.append(deriv_row)

    if excel_data:
        df = pd.DataFrame(excel_data)

        # 创建统计数据
        # 1. 每个 base model 的衍生模型数量统计（按类型）
        stats_data = []
        for base_model_id, result in all_results.items():
            if result['derivatives']:
                # 统计各类型数量
                type_counts = {}
                for deriv in result['derivatives']:
                    model_type = deriv.get('model_type', 'other')
                    type_counts[model_type] = type_counts.get(model_type, 0) + 1

                # 计算总下载量
                total_downloads = sum(deriv.get('download_count', 0) for deriv in result['derivatives'])

                stats_row = {
                    'base_model': base_model_id,
                    'total_derivatives': len(result['derivatives']),
                    'quantized': type_counts.get('quantized', 0),
                    'finetune': type_counts.get('finetune', 0),
                    'adapter': type_counts.get('adapter', 0),
                    'lora': type_counts.get('lora', 0),
                    'merge': type_counts.get('merge', 0),
                    'other': type_counts.get('other', 0),
                    'total_derivative_downloads': total_downloads,
                    'avg_downloads_per_derivative': total_downloads / len(result['derivatives']) if result['derivatives'] else 0
                }
                stats_data.append(stats_row)

        stats_df = pd.DataFrame(stats_data)

        # 保存到 Excel，包含多个 sheet
        excel_filename = f"qwen_model_tree_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Sheet 1: 统计汇总
            stats_df.to_excel(writer, sheet_name='统计汇总', index=False)

            # Sheet 2: 所有模型详细信息
            df.to_excel(writer, sheet_name='详细数据', index=False)

            # Sheet 3-N: 每个 base model 单独一个 sheet
            for base_model_id, result in all_results.items():
                sheet_name = base_model_id.split('/')[-1][:31]  # Excel sheet 名称限制 31 字符

                # 该 base model 的所有数据（包括 base model 自己和衍生模型）
                base_df = df[df['base_model'] == base_model_id].copy()
                base_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"📊 Excel 数据已保存到: {excel_filename}")
        print(f"   - 统计汇总: 每个 base model 的衍生模型类型统计")
        print(f"   - 详细数据: 所有模型的完整信息")
        print(f"   - 各 base model 独立 sheet: {len(all_results)} 个")

        # 打印总体统计
        print(f"\n{'='*80}")
        print(f"📊 总体统计")
        print(f"{'='*80}")
        print(f"基础模型数: {len(QWEN_MODELS)}")
        print(f"衍生模型总数: {len(df[~df['is_base']])}")
        print(f"总记录数: {len(df)}")

        if not df[~df['is_base']].empty:
            print(f"\n衍生模型按类型分布:")
            type_counts = df[~df['is_base']]['model_type'].value_counts()
            for model_type, count in type_counts.items():
                print(f"  {model_type}: {count}")

            print(f"\n衍生模型按基础模型分布:")
            base_counts = df[~df['is_base']]['base_model'].value_counts()
            for base_model, count in base_counts.items():
                print(f"  {base_model}: {count}")

            print(f"\n总下载量统计:")
            total_downloads = df['download_count'].sum()
            base_downloads = df[df['is_base']]['download_count'].sum()
            deriv_downloads = df[~df['is_base']]['download_count'].sum()
            print(f"  基础模型总下载量: {base_downloads:,}")
            print(f"  衍生模型总下载量: {deriv_downloads:,}")
            print(f"  总计: {total_downloads:,}")

            # 详细统计表格
            print(f"\n{'='*80}")
            print(f"📋 各基础模型的衍生模型类型统计")
            print(f"{'='*80}")
            print(stats_df.to_string(index=False))

    print(f"\n✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ 所有数据已保存")


if __name__ == "__main__":
    main()
