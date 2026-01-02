"""
获取 Qwen 模型的 Model Tree 数据 - 版本2
支持 Qwen3 和 Qwen3-VL 分开统计，并按模型分组
"""
from huggingface_hub import list_models, model_info
from datetime import datetime
import pandas as pd
import json
import re
from typing import List, Dict
import sys
sys.path.append('/Users/zhanghaoxin/Desktop/Baidu/DownloadData')
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
    """获取模型的所有信息字段"""
    try:
        info = model_info(model_id, expand=["downloadsAllTime", "trendingScore"])
        card_data = None
        if hasattr(info, 'cardData') and info.cardData:
            if isinstance(info.cardData, dict):
                card_data = info.cardData
            elif hasattr(info.cardData, '__dict__'):
                card_data = info.cardData.__dict__

        model_obj = None
        try:
            models = list(list_models(model_name=model_id, full=True, limit=1))
            if models:
                model_obj = models[0]
        except:
            pass

        model_data = {}

        # 基本字段
        if hasattr(info, 'modelId'):
            model_data['modelId'] = info.modelId
        if hasattr(info, 'author'):
            model_data['author'] = info.author
        if hasattr(info, 'downloads_all_time'):
            model_data['downloads_all_time'] = info.downloads_all_time
        if hasattr(info, 'downloads'):
            model_data['downloads'] = info.downloads
        if hasattr(info, 'likes'):
            model_data['likes'] = info.likes
        if hasattr(info, 'library_name'):
            model_data['library_name'] = info.library_name
        if hasattr(info, 'pipeline_tag'):
            model_data['pipeline_tag'] = info.pipeline_tag
        if hasattr(info, 'created_at'):
            model_data['created_at'] = info.created_at.isoformat() if hasattr(info.created_at, 'isoformat') else str(info.created_at)
        if hasattr(info, 'last_modified'):
            model_data['last_modified'] = info.last_modified.isoformat() if hasattr(info.last_modified, 'isoformat') else str(info.last_modified)

        # 优先使用 model_info 提供的 tags，其次 fallback 到 list_models 返回的 tags
        if hasattr(info, 'tags') and info.tags:
            model_data['tags'] = info.tags
        elif model_obj and hasattr(model_obj, 'tags'):
            model_data['tags'] = model_obj.tags
        else:
            model_data['tags'] = []

        if model_obj and hasattr(model_obj, 'trending_score'):
            model_data['trending_score'] = model_obj.trending_score

        # 分类字段
        publisher = model_data.get('author', 'Unknown')
        tags = model_data.get('tags', [])
        pipeline_tag = model_data.get('pipeline_tag', None)

        model_data['model_category'] = classify_model(model_id, publisher)
        model_data['model_type'] = classify_model_type(model_id, tags, pipeline_tag, card_data)
        model_data['download_count'] = model_data.get('downloads_all_time') or model_data.get('downloads', 0) or 0
        model_data['fetched_at'] = datetime.now().isoformat()

        return model_data

    except Exception as e:
        print(f"    ❌ 获取模型 {model_id} 信息失败: {e}")
        return None


def get_model_tree_with_full_info(base_model_id: str) -> Dict:
    """获取指定模型的完整 Model Tree 信息"""
    print(f"\n{'='*80}")
    print(f"📊 获取 {base_model_id} 的 Model Tree")
    print(f"{'='*80}")

    result = {
        'base_model_id': base_model_id,
        'base_model_info': None,
        'derivatives': []
    }

    # 获取基础模型信息
    print(f"\n1️⃣ 获取基础模型信息...")
    base_info = get_all_model_info_fields(base_model_id)
    if not base_info:
        print(f"  ❌ 无法获取基础模型 {base_model_id} 的信息")
        return result

    result['base_model_info'] = base_info
    print(f"  ✅ 基础模型信息获取成功 | 下载量: {base_info.get('download_count', 0):,}")

    # 获取衍生模型
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

    # 获取衍生模型详情
    print(f"\n3️⃣ 获取衍生模型详细信息...")
    for idx, deriv in enumerate(derivatives, 1):
        print(f"  [{idx}/{len(derivatives)}] {deriv.id}")

        deriv_info = get_all_model_info_fields(deriv.id)
        if deriv_info:
            # 确保字段正确
            if 'modelId' not in deriv_info or not deriv_info['modelId']:
                deriv_info['modelId'] = deriv.id
            if 'author' not in deriv_info or not deriv_info['author']:
                deriv_info['author'] = deriv.author if hasattr(deriv, 'author') else ''

            deriv_info['base_model'] = base_model_id
            result['derivatives'].append(deriv_info)

            print(f"     ✅ 下载量: {deriv_info.get('download_count', 0):,} | 类型: {deriv_info.get('model_type', 'N/A')}")

    return result


def main():
    """主函数"""
    print("🚀 开始获取 Qwen 模型的 Model Tree 数据")
    print(f"Qwen3 模型数: {len(QWEN3_MODELS)}")
    print(f"Qwen3-VL 模型数: {len(QWEN3_VL_MODELS)}")
    print(f"总计: {len(QWEN3_MODELS) + len(QWEN3_VL_MODELS)} 个模型")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 分别获取两个系列
    qwen3_results = {}
    qwen3_vl_results = {}

    print(f"\n{'#'*80}")
    print("📦 获取 Qwen3 系列")
    print(f"{'#'*80}")
    for model_id in QWEN3_MODELS:
        result = get_model_tree_with_full_info(model_id)
        qwen3_results[model_id] = result

    print(f"\n{'#'*80}")
    print("📦 获取 Qwen3-VL 系列")
    print(f"{'#'*80}")
    for model_id in QWEN3_VL_MODELS:
        result = get_model_tree_with_full_info(model_id)
        qwen3_vl_results[model_id] = result

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建 DataFrame
    def create_dataframe(results_dict):
        """将结果字典转换为 DataFrame"""
        data = []
        for base_model_id, result in results_dict.items():
            # 添加基础模型
            if result['base_model_info']:
                base_row = {
                    'model_id': base_model_id,
                    'base_model': base_model_id,
                    'model_group': extract_model_group(base_model_id),
                    'is_base': True,
                    'model_name': base_model_id.split('/')[-1],
                    'publisher': result['base_model_info'].get('author', ''),
                    'model_type': result['base_model_info'].get('model_type', ''),
                    'download_count': result['base_model_info'].get('download_count', 0),
                    'likes': result['base_model_info'].get('likes', 0),
                    'library_name': result['base_model_info'].get('library_name', ''),
                    'pipeline_tag': result['base_model_info'].get('pipeline_tag', ''),
                    'created_at': result['base_model_info'].get('created_at', ''),
                    'last_modified': result['base_model_info'].get('last_modified', ''),
                    'fetched_at': result['base_model_info'].get('fetched_at', '')
                }
                data.append(base_row)

            # 添加衍生模型
            for deriv_info in result['derivatives']:
                deriv_row = {
                    'model_id': deriv_info.get('modelId', ''),
                    'base_model': base_model_id,
                    'model_group': extract_model_group(base_model_id),
                    'is_base': False,
                    'model_name': deriv_info.get('modelId', '').split('/')[-1] if deriv_info.get('modelId') else '',
                    'publisher': deriv_info.get('author', ''),
                    'model_type': deriv_info.get('model_type', ''),
                    'download_count': deriv_info.get('download_count', 0),
                    'likes': deriv_info.get('likes', 0),
                    'library_name': deriv_info.get('library_name', ''),
                    'pipeline_tag': deriv_info.get('pipeline_tag', ''),
                    'created_at': deriv_info.get('created_at', ''),
                    'last_modified': deriv_info.get('last_modified', ''),
                    'fetched_at': deriv_info.get('fetched_at', '')
                }
                data.append(deriv_row)

        return pd.DataFrame(data)

    # 创建两个 DataFrame
    df_qwen3 = create_dataframe(qwen3_results)
    df_qwen3_vl = create_dataframe(qwen3_vl_results)

    # 创建统计汇总
    def create_stats(df, series_name):
        """创建统计汇总表"""
        stats_data = []

        # 按 model_group 分组统计
        for group_name in df[df['is_base'] == True]['model_group'].unique():
            # 获取该分组的所有衍生模型
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

        return pd.DataFrame(stats_data)

    stats_qwen3 = create_stats(df_qwen3, 'Qwen3')
    stats_qwen3_vl = create_stats(df_qwen3_vl, 'Qwen3-VL')
    stats_combined = pd.concat([stats_qwen3, stats_qwen3_vl], ignore_index=True)

    # 保存到 Excel（3个 sheet）
    excel_filename = f"qwen_model_tree_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Sheet 1: 统计汇总
        stats_combined.to_excel(writer, sheet_name='统计汇总', index=False)
        # Sheet 2: Qwen3
        df_qwen3.to_excel(writer, sheet_name='Qwen3', index=False)
        # Sheet 3: Qwen3-VL
        df_qwen3_vl.to_excel(writer, sheet_name='Qwen3-VL', index=False)

    print(f"\n✅ Excel 数据已保存到: {excel_filename}")
    print(f"   - 统计汇总: 按模型分组的统计（包含数量和下载量百分比）")
    print(f"   - Qwen3: {len(df_qwen3)} 条记录")
    print(f"   - Qwen3-VL: {len(df_qwen3_vl)} 条记录")

    print(f"\n✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
