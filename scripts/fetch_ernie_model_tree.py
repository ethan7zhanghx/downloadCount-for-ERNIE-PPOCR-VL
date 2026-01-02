"""
获取 ERNIE 模型的 Model Tree 和 Search 数据
支持 ERNIE-4.5 和 PaddleOCR-VL 分开统计，并按模型分组
同时包含 Model Tree 和 Search 获取的衍生模型
"""
from huggingface_hub import list_models, model_info
from datetime import datetime
import pandas as pd
import json
import re
from typing import List, Dict, Set
import sys
sys.path.append('/Users/zhanghaoxin/Desktop/Baidu/DownloadData')
from ernie_tracker.fetchers.fetchers_modeltree import classify_model, classify_model_type


# ERNIE-4.5 官方模型列表
ERNIE_45_MODELS = [
    # 0.3B 系列
    "baidu/ERNIE-4.5-0.3B-PT",
    "baidu/ERNIE-4.5-0.3B-Paddle",
    "baidu/ERNIE-4.5-0.3B-Base-PT",
    "baidu/ERNIE-4.5-0.3B-Base-Paddle",
    # 21B-A3B 系列
    "baidu/ERNIE-4.5-21B-A3B-PT",
    "baidu/ERNIE-4.5-21B-A3B-Paddle",
    "baidu/ERNIE-4.5-21B-A3B-Base-PT",
    "baidu/ERNIE-4.5-21B-A3B-Base-Paddle",
    "baidu/ERNIE-4.5-21B-A3B-Thinking",
    # 300B-A47B 系列
    "baidu/ERNIE-4.5-300B-A47B-PT",
    "baidu/ERNIE-4.5-300B-A47B-Paddle",
    "baidu/ERNIE-4.5-300B-A47B-Base-PT",
    "baidu/ERNIE-4.5-300B-A47B-Base-Paddle",
    "baidu/ERNIE-4.5-300B-A47B-FP8-Paddle",
    "baidu/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle",
    "baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle",
    "baidu/ERNIE-4.5-300B-A47B-2Bits-TP2-Paddle",
    "baidu/ERNIE-4.5-300B-A47B-2Bits-TP4-Paddle",
    # VL-28B-A3B 系列
    "baidu/ERNIE-4.5-VL-28B-A3B-PT",
    "baidu/ERNIE-4.5-VL-28B-A3B-Paddle",
    "baidu/ERNIE-4.5-VL-28B-A3B-Base-PT",
    "baidu/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
    "baidu/ERNIE-4.5-VL-28B-A3B-Thinking",
    # VL-424B-A47B 系列
    "baidu/ERNIE-4.5-VL-424B-A47B-PT",
    "baidu/ERNIE-4.5-VL-424B-A47B-Paddle",
    "baidu/ERNIE-4.5-VL-424B-A47B-Base-PT",
    "baidu/ERNIE-4.5-VL-424B-A47B-Base-Paddle",
]

# PaddleOCR-VL 官方模型列表
PADDLEOCR_VL_MODELS = [
    "PaddlePaddle/PaddleOCR-VL",
]


def extract_model_group(model_id: str) -> str:
    """
    提取模型分组名称：找到最后一次出现的"数字+B"，之后的内容去掉

    例如：
    - baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle → ERNIE-4.5-300B-A47B
    - baidu/ERNIE-4.5-21B-A3B-Thinking → ERNIE-4.5-21B-A3B
    - baidu/ERNIE-4.5-0.3B-PT → ERNIE-4.5-0.3B

    Args:
        model_id: 完整的模型 ID

    Returns:
        str: 分组名称
    """
    # 去掉发布者前缀
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id

    # 匹配所有的"数字+B"模式（包括小数和 A47B/A3B 这种格式）
    # 匹配模式：可选的 A + 数字（可能包含小数点） + B
    # 例如：300B, 21B, A47B, A3B, 0.3B
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
    """获取模型的所有信息字段，包括尝试获取 base_model"""
    try:
        info = model_info(model_id, expand=["downloadsAllTime", "trendingScore"])

        model_obj = None
        try:
            models = list(list_models(model_name=model_id, full=True, limit=1))
            if models:
                model_obj = models[0]
        except:
            pass

        model_data = {}
        card_data = None
        if hasattr(info, 'cardData') and info.cardData:
            if isinstance(info.cardData, dict):
                card_data = info.cardData
            elif hasattr(info.cardData, '__dict__'):
                card_data = info.cardData.__dict__

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

        # 尝试从 API 获取 base_model
        if card_data:
            if 'base_model' in card_data:
                base_model_from_api = card_data['base_model']
                # base_model 可能是字符串或列表
                if isinstance(base_model_from_api, list) and len(base_model_from_api) > 0:
                    model_data['base_model_from_api'] = base_model_from_api[0]
                elif isinstance(base_model_from_api, str) and base_model_from_api:
                    model_data['base_model_from_api'] = base_model_from_api

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


def search_models_with_keyword(keyword: str, exclude_ids: Set[str] = None) -> List[Dict]:
    """
    通过关键词搜索模型

    Args:
        keyword: 搜索关键词
        exclude_ids: 需要排除的模型 ID 集合（已在 Model Tree 中获取的）

    Returns:
        List[Dict]: 搜索到的模型信息列表
    """
    if exclude_ids is None:
        exclude_ids = set()

    print(f"\n🔍 搜索包含 '{keyword}' 的模型...")

    try:
        search_results = list(list_models(
            search=keyword,
            full=True,
            limit=1000,
            sort="downloads",
            direction=-1
        ))

        print(f"  ✅ 搜索到 {len(search_results)} 个模型")

        # 过滤掉已经在 Model Tree 中的模型
        filtered_results = []
        for model in search_results:
            if model.id not in exclude_ids:
                filtered_results.append(model)

        print(f"  ✅ 去重后剩余 {len(filtered_results)} 个新模型")
        return filtered_results

    except Exception as e:
        print(f"  ❌ 搜索失败: {e}")
        return []


def get_model_tree_and_search(base_model_id: str, search_keywords: List[str] = None) -> Dict:
    """
    获取指定模型的完整 Model Tree 和 Search 信息

    Args:
        base_model_id: 基础模型 ID
        search_keywords: 用于搜索衍生模型的关键词列表

    Returns:
        Dict: 包含基础模型信息和衍生模型列表的字典
    """
    print(f"\n{'='*80}")
    print(f"📊 获取 {base_model_id} 的 Model Tree 和 Search 数据")
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

    # 用于记录已获取的模型 ID
    seen_model_ids = {base_model_id}

    # 获取 Model Tree 衍生模型
    print(f"\n2️⃣ 查找 Model Tree 衍生模型...")
    model_tree_derivatives = []
    try:
        derivatives = list(list_models(
            filter=f"base_model:{base_model_id}",
            full=True,
            limit=1000
        ))

        if derivatives:
            print(f"  ✅ 找到 {len(derivatives)} 个 Model Tree 衍生模型")
            model_tree_derivatives = derivatives
        else:
            print(f"  ⚪ 没有找到 Model Tree 衍生模型")
    except Exception as e:
        print(f"  ❌ 查找 Model Tree 衍生模型失败: {e}")

    # 获取 Model Tree 衍生模型详情
    if model_tree_derivatives:
        print(f"\n3️⃣ 获取 Model Tree 衍生模型详细信息...")
        for idx, deriv in enumerate(model_tree_derivatives, 1):
            print(f"  [{idx}/{len(model_tree_derivatives)}] {deriv.id}")

            deriv_info = get_all_model_info_fields(deriv.id)
            if deriv_info:
                # 确保字段正确
                if 'modelId' not in deriv_info or not deriv_info['modelId']:
                    deriv_info['modelId'] = deriv.id
                if 'author' not in deriv_info or not deriv_info['author']:
                    deriv_info['author'] = deriv.author if hasattr(deriv, 'author') else ''

                deriv_info['base_model'] = base_model_id
                deriv_info['data_source'] = 'model_tree'
                result['derivatives'].append(deriv_info)
                seen_model_ids.add(deriv.id)

                print(f"     ✅ 下载量: {deriv_info.get('download_count', 0):,} | 类型: {deriv_info.get('model_type', 'N/A')}")

    # 通过 Search 查找衍生模型
    if search_keywords:
        print(f"\n4️⃣ 通过关键词搜索衍生模型...")
        for keyword in search_keywords:
            search_results = search_models_with_keyword(keyword, exclude_ids=seen_model_ids)

            if search_results:
                print(f"\n5️⃣ 获取 Search 衍生模型详细信息 (关键词: {keyword})...")
                for idx, model in enumerate(search_results, 1):
                    # 跳过基础模型本身
                    if model.id == base_model_id:
                        continue

                    # 跳过已处理的模型
                    if model.id in seen_model_ids:
                        continue

                    print(f"  [{idx}/{len(search_results)}] {model.id}")

                    deriv_info = get_all_model_info_fields(model.id)
                    if deriv_info:
                        # 确保字段正确
                        if 'modelId' not in deriv_info or not deriv_info['modelId']:
                            deriv_info['modelId'] = model.id
                        if 'author' not in deriv_info or not deriv_info['author']:
                            deriv_info['author'] = model.author if hasattr(model, 'author') else ''

                        # Search 获取的模型可能没有 base_model，尝试推断
                        deriv_info['base_model'] = base_model_id
                        deriv_info['data_source'] = 'search'
                        result['derivatives'].append(deriv_info)
                        seen_model_ids.add(model.id)

                        print(f"     ✅ 下载量: {deriv_info.get('download_count', 0):,} | 类型: {deriv_info.get('model_type', 'N/A')}")

    print(f"\n📊 汇总:")
    print(f"  总衍生模型数: {len(result['derivatives'])}")
    model_tree_count = sum(1 for d in result['derivatives'] if d.get('data_source') == 'model_tree')
    search_count = sum(1 for d in result['derivatives'] if d.get('data_source') == 'search')
    print(f"  - Model Tree: {model_tree_count}")
    print(f"  - Search: {search_count}")

    return result


def main():
    """主函数"""
    print("🚀 开始获取 ERNIE 模型数据")
    print(f"ERNIE-4.5 模型数: {len(ERNIE_45_MODELS)}")
    print(f"PaddleOCR-VL 模型数: {len(PADDLEOCR_VL_MODELS)}")
    print(f"总计: {len(ERNIE_45_MODELS) + len(PADDLEOCR_VL_MODELS)} 个模型")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========== ERNIE-4.5 系列 ==========
    print(f"\n{'#'*80}")
    print("📦 处理 ERNIE-4.5 系列")
    print(f"{'#'*80}")

    # 1. 全局搜索 "ERNIE-4.5"
    print(f"\n{'='*80}")
    print(f"🔍 步骤 1: 全局搜索 'ERNIE-4.5'")
    print(f"{'='*80}")
    ernie_45_search_results = search_models_with_keyword("ERNIE-4.5", exclude_ids=set())

    # 用于存储所有模型信息
    ernie_45_all_models = {}

    # 添加搜索到的模型
    print(f"\n获取搜索模型详细信息...")
    for idx, model in enumerate(ernie_45_search_results, 1):
        print(f"  [{idx}/{len(ernie_45_search_results)}] {model.id}")
        model_data = get_all_model_info_fields(model.id)
        if model_data:
            model_data['modelId'] = model.id
            model_data['author'] = model.author if hasattr(model, 'author') else model_data.get('author', '')
            model_data['data_source'] = 'search'

            # 尝试设置 base_model
            if 'base_model_from_api' in model_data and model_data['base_model_from_api']:
                model_data['base_model'] = model_data['base_model_from_api']
                print(f"     ✅ base_model (从API): {model_data['base_model']}")

            ernie_45_all_models[model.id] = model_data
            print(f"     ✅ 下载量: {model_data.get('download_count', 0):,} | created_at: {model_data.get('created_at', 'N/A')}")

    # 2. 获取官方模型的 Model Tree
    print(f"\n{'='*80}")
    print(f"📊 步骤 2: 获取官方模型的 Model Tree")
    print(f"{'='*80}")

    for model_id in ERNIE_45_MODELS:
        print(f"\n处理官方模型: {model_id}")

        # 如果官方模型不在搜索结果中，单独获取
        if model_id not in ernie_45_all_models:
            print(f"  获取官方模型信息...")
            base_info = get_all_model_info_fields(model_id)
            if base_info:
                base_info['modelId'] = model_id
                base_info['data_source'] = 'original'
                ernie_45_all_models[model_id] = base_info
                print(f"  ✅ 官方模型信息获取成功")
        else:
            # 标记为官方模型
            ernie_45_all_models[model_id]['data_source'] = 'original'
            print(f"  ✅ 官方模型已在搜索结果中")

        # 获取该官方模型的 Model Tree
        print(f"  获取 Model Tree...")
        try:
            derivatives = list(list_models(
                filter=f"base_model:{model_id}",
                full=True,
                limit=1000
            ))

            if derivatives:
                print(f"  ✅ 找到 {len(derivatives)} 个 Model Tree 衍生模型")
                for idx, deriv in enumerate(derivatives, 1):
                    if deriv.id not in ernie_45_all_models:
                        # 新模型，完整获取信息
                        print(f"    [{idx}/{len(derivatives)}] {deriv.id}")
                        deriv_info = get_all_model_info_fields(deriv.id)
                        if deriv_info:
                            deriv_info['modelId'] = deriv.id
                            deriv_info['author'] = deriv.author if hasattr(deriv, 'author') else deriv_info.get('author', '')
                            deriv_info['base_model'] = model_id  # 保存 base_model
                            deriv_info['data_source'] = 'model_tree'
                            ernie_45_all_models[deriv.id] = deriv_info
                            print(f"       ✅ 下载量: {deriv_info.get('download_count', 0):,} | created_at: {deriv_info.get('created_at', 'N/A')}")
                    else:
                        # 模型已存在（之前通过 search 添加的），更新 base_model 和 data_source
                        print(f"    [{idx}/{len(derivatives)}] {deriv.id} (已存在，更新 base_model)")
                        ernie_45_all_models[deriv.id]['base_model'] = model_id
                        ernie_45_all_models[deriv.id]['data_source'] = 'both'  # 同时在 search 和 model_tree 里
            else:
                print(f"  ⚪ 没有找到 Model Tree 衍生模型")
        except Exception as e:
            print(f"  ❌ 获取 Model Tree 失败: {e}")

    # ========== PaddleOCR-VL 系列 ==========
    print(f"\n{'#'*80}")
    print("📦 处理 PaddleOCR-VL 系列")
    print(f"{'#'*80}")

    # 1. 全局搜索 "PaddleOCR-VL"
    print(f"\n{'='*80}")
    print(f"🔍 步骤 1: 全局搜索 'PaddleOCR-VL'")
    print(f"{'='*80}")
    paddleocr_vl_search_results = search_models_with_keyword("PaddleOCR-VL", exclude_ids=set())

    paddleocr_vl_all_models = {}

    # 添加搜索到的模型
    print(f"\n获取搜索模型详细信息...")
    for idx, model in enumerate(paddleocr_vl_search_results, 1):
        print(f"  [{idx}/{len(paddleocr_vl_search_results)}] {model.id}")
        model_data = get_all_model_info_fields(model.id)
        if model_data:
            model_data['modelId'] = model.id
            model_data['author'] = model.author if hasattr(model, 'author') else model_data.get('author', '')
            model_data['data_source'] = 'search'

            # 尝试设置 base_model
            if 'base_model_from_api' in model_data and model_data['base_model_from_api']:
                model_data['base_model'] = model_data['base_model_from_api']
                print(f"     ✅ base_model (从API): {model_data['base_model']}")

            paddleocr_vl_all_models[model.id] = model_data
            print(f"     ✅ 下载量: {model_data.get('download_count', 0):,} | created_at: {model_data.get('created_at', 'N/A')}")

    # 2. 获取官方模型的 Model Tree
    print(f"\n{'='*80}")
    print(f"📊 步骤 2: 获取官方模型的 Model Tree")
    print(f"{'='*80}")

    for model_id in PADDLEOCR_VL_MODELS:
        print(f"\n处理官方模型: {model_id}")

        # 如果官方模型不在搜索结果中，单独获取
        if model_id not in paddleocr_vl_all_models:
            print(f"  获取官方模型信息...")
            base_info = get_all_model_info_fields(model_id)
            if base_info:
                base_info['modelId'] = model_id
                base_info['data_source'] = 'original'
                paddleocr_vl_all_models[model_id] = base_info
                print(f"  ✅ 官方模型信息获取成功")
        else:
            # 标记为官方模型
            paddleocr_vl_all_models[model_id]['data_source'] = 'original'
            print(f"  ✅ 官方模型已在搜索结果中")

        # 获取该官方模型的 Model Tree
        print(f"  获取 Model Tree...")
        try:
            derivatives = list(list_models(
                filter=f"base_model:{model_id}",
                full=True,
                limit=1000
            ))

            if derivatives:
                print(f"  ✅ 找到 {len(derivatives)} 个 Model Tree 衍生模型")
                for idx, deriv in enumerate(derivatives, 1):
                    if deriv.id not in paddleocr_vl_all_models:
                        # 新模型，完整获取信息
                        print(f"    [{idx}/{len(derivatives)}] {deriv.id}")
                        deriv_info = get_all_model_info_fields(deriv.id)
                        if deriv_info:
                            deriv_info['modelId'] = deriv.id
                            deriv_info['author'] = deriv.author if hasattr(deriv, 'author') else deriv_info.get('author', '')
                            deriv_info['base_model'] = model_id  # 保存 base_model
                            deriv_info['data_source'] = 'model_tree'
                            paddleocr_vl_all_models[deriv.id] = deriv_info
                            print(f"       ✅ 下载量: {deriv_info.get('download_count', 0):,} | created_at: {deriv_info.get('created_at', 'N/A')}")
                    else:
                        # 模型已存在（之前通过 search 添加的），更新 base_model 和 data_source
                        print(f"    [{idx}/{len(derivatives)}] {deriv.id} (已存在，更新 base_model)")
                        paddleocr_vl_all_models[deriv.id]['base_model'] = model_id
                        paddleocr_vl_all_models[deriv.id]['data_source'] = 'both'  # 同时在 search 和 model_tree 里
            else:
                print(f"  ⚪ 没有找到 Model Tree 衍生模型")
        except Exception as e:
            print(f"  ❌ 获取 Model Tree 失败: {e}")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建 DataFrame
    def create_dataframe(all_models_dict, official_model_ids):
        """
        将模型字典转换为 DataFrame

        Args:
            all_models_dict: 所有模型信息的字典 {model_id: model_info}
            official_model_ids: 官方模型 ID 列表
        """
        data = []

        for model_id, model_info in all_models_dict.items():
            # 判断是否是官方基础模型
            is_base = model_id in official_model_ids

            # 尝试从模型信息中推断 base_model
            base_model = model_id if is_base else ''

            # 对于衍生模型，尝试从名称或其他信息推断 base_model
            if not is_base:
                # 检查是否有明确的 base_model 字段（从 Model Tree 获取的）
                if 'base_model' in model_info and model_info['base_model']:
                    base_model = model_info['base_model']
                else:
                    # 尝试从模型名称推断
                    for official_id in official_model_ids:
                        official_name = official_id.split('/')[-1]
                        if official_name in model_id:
                            base_model = official_id
                            break

            row = {
                'model_id': model_id,
                'base_model': base_model,
                'model_group': extract_model_group(base_model) if base_model else extract_model_group(model_id),
                'is_base': is_base,
                'data_source': model_info.get('data_source', 'unknown'),
                'model_name': model_id.split('/')[-1],
                'publisher': model_info.get('author', ''),
                'model_type': model_info.get('model_type', ''),
                'download_count': model_info.get('download_count', 0),
                'likes': model_info.get('likes', 0),
                'library_name': model_info.get('library_name', ''),
                'pipeline_tag': model_info.get('pipeline_tag', ''),
                'created_at': model_info.get('created_at', ''),
                'last_modified': model_info.get('last_modified', ''),
                'fetched_at': model_info.get('fetched_at', '')
            }
            data.append(row)

        return pd.DataFrame(data)

    # 创建两个 DataFrame
    df_ernie_45 = create_dataframe(ernie_45_all_models, ERNIE_45_MODELS)
    df_paddleocr_vl = create_dataframe(paddleocr_vl_all_models, PADDLEOCR_VL_MODELS)

    # 创建统计汇总
    def create_stats(df, series_name):
        """创建统计汇总表"""
        stats_data = []

        # 按 model_group 分组统计（包括所有 model_group，不只是官方的）
        all_model_groups = df[df['is_base'] == False]['model_group'].unique()

        for group_name in all_model_groups:
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

        # 按 total_downloads 降序排序
        stats_df = pd.DataFrame(stats_data)
        if len(stats_df) > 0:
            stats_df = stats_df.sort_values('total_downloads', ascending=False)

        return stats_df

    stats_ernie_45 = create_stats(df_ernie_45, 'ERNIE-4.5')
    stats_paddleocr_vl = create_stats(df_paddleocr_vl, 'PaddleOCR-VL')
    stats_combined = pd.concat([stats_ernie_45, stats_paddleocr_vl], ignore_index=True)

    # 保存到 Excel（3个 sheet）
    excel_filename = f"ernie_model_tree_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Sheet 1: 统计汇总
        stats_combined.to_excel(writer, sheet_name='统计汇总', index=False)
        # Sheet 2: ERNIE-4.5
        df_ernie_45.to_excel(writer, sheet_name='ERNIE-4.5', index=False)
        # Sheet 3: PaddleOCR-VL
        df_paddleocr_vl.to_excel(writer, sheet_name='PaddleOCR-VL', index=False)

    print(f"\n✅ Excel 数据已保存到: {excel_filename}")
    print(f"   - 统计汇总: 按模型分组的统计（包含数量和下载量百分比）")
    print(f"   - ERNIE-4.5: {len(df_ernie_45)} 条记录")
    print(f"   - PaddleOCR-VL: {len(df_paddleocr_vl)} 条记录")

    # 统计 data_source 分布
    print(f"\n📊 数据来源统计:")
    print(f"   ERNIE-4.5:")
    if len(df_ernie_45) > 0:
        for source in ['original', 'model_tree', 'search']:
            count = len(df_ernie_45[df_ernie_45['data_source'] == source])
            if count > 0:
                print(f"     - {source}: {count} 条")
    print(f"   PaddleOCR-VL:")
    if len(df_paddleocr_vl) > 0:
        for source in ['original', 'model_tree', 'search']:
            count = len(df_paddleocr_vl[df_paddleocr_vl['data_source'] == source])
            if count > 0:
                print(f"     - {source}: {count} 条")

    print(f"\n✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
