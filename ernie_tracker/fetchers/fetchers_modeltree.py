"""
Model Tree 功能模块 - 获取官方模型的衍生模型
支持获取 Finetune 和 Adapter 模型，并智能分类
"""
from huggingface_hub import list_models, model_info
from datetime import date
import pandas as pd
import time
import re
from typing import List, Dict, Set, Tuple
from ..db import save_to_db, get_last_model_count, update_last_model_count
from ..config import DB_PATH


def classify_model(model_name: str, publisher: str, base_model: str = None) -> str:
    """
    智能分类模型

    Args:
        model_name: 模型名称
        publisher: 发布者
        base_model: 基础模型ID（可选，用于衍生模型的分类）

    Returns:
        str: 模型类别 ('ernie-4.5', 'paddleocr-vl', 'other-ernie', 'other')
    """
    """
    仅返回两类：ernie-4.5 或 paddleocr-vl。其余一律归入 ernie-4.5（避免出现 other/other-ernie）。
    """
    def _is_paddleocr(name: str) -> bool:
        n = name.lower()
        return 'paddleocr' in n and 'vl' in n

    base_lower = base_model.lower() if base_model else ''
    name_lower = model_name.lower()

    if _is_paddleocr(base_lower) or _is_paddleocr(name_lower):
        return 'paddleocr-vl'

    # 默认归入 ernie-4.5
    return 'ernie-4.5'


def classify_model_type(model_name: str, tags: list, pipeline_tag: str = None, card_data: dict = None) -> str:
    """
    识别模型类型，优先使用结构化信息（HF 标签 / 模型卡），最后再名称兜底

    Args:
        model_name: 模型名称
        tags: 模型标签列表（来自HuggingFace API）
        pipeline_tag: pipeline标签
        card_data: 模型卡的元信息（如果可用）

    Returns:
        str: 模型类型
        - 'quantized': 量化模型
        - 'finetune': 微调模型
        - 'adapter': Adapter模型
        - 'lora': LoRA模型
        - 'merge': 合并模型
        - 'original': 官方原始模型
        - 'other': 其他
    """
    card_data_dict = card_data if isinstance(card_data, dict) else None

    # 1) 标签：结构化、优先级最高
    tags_lower = [tag.lower() for tag in tags] if tags else []
    for tag in tags_lower:
        if tag.startswith('base_model:quantized:'):
            return 'quantized'
        if tag.startswith('base_model:adapter:'):
            return 'adapter'
        if tag.startswith('base_model:lora:'):
            return 'lora'
        if tag.startswith('base_model:merge:'):
            return 'merge'
        if tag.startswith('base_model:finetune:'):
            return 'finetune'

    # 2) 标签：PEFT 信号
    peft_indicators = ['peft', 'prefix-tuning', 'prompt-tuning', 'adapter']
    if tags_lower and any(indicator in ' '.join(tags_lower) for indicator in peft_indicators):
        if any('lora' in tag for tag in tags_lower):
            return 'lora'
        return 'adapter'

    # 3) 模型卡：仅使用常见、固定字段
    if card_data_dict:
        card_type = _classify_by_card_data(card_data_dict)
        if card_type != 'other':
            return card_type

    # 4) 官方原始模型（无 base_model 标签）
    official_patterns = ['baidu/', 'paddlepaddle/']
    model_name_lower = model_name.lower()
    if any(pattern in model_name_lower for pattern in official_patterns):
        if not any(tag.startswith('base_model:') for tag in tags_lower):
            return 'original'

    # 5) 名称兜底（最不可靠）
    name_based_type = _classify_by_name_fallback(model_name)
    if name_based_type != 'other':
        return name_based_type

    return 'other'


def _classify_by_card_data(card_data: dict) -> str:
    """
    基于模型卡的字段进行分类（优先级最高）
    """
    # 量化相关字段
    quant_keys = [
        'quantization_config', 'quantization', 'quantization_bits',
        'load_in_4bit', 'load_in_8bit', 'bnb_4bit_quant_type', 'gguf'
    ]
    if any(key in card_data for key in quant_keys):
        return 'quantized'

    # PEFT / LoRA
    peft_type = str(card_data.get('peft_type', '') or '').lower()
    if 'lora' in peft_type:
        return 'lora'
    if peft_type:
        return 'adapter'

    if any(key in card_data for key in ['lora_alpha', 'lora_r', 'lora_dropout']):
        return 'lora'
    if any(key in card_data for key in ['adapters', 'adapter', 'adapter_config', 'adapter_name']):
        return 'adapter'

    # 合并模型
    if any(key in card_data for key in ['merge_method', 'merging_config', 'merge_config', 'merged_by']):
        return 'merge'

    # 明确的微调配置字段
    finetune_keys = ['finetuning_type', 'finetuning_config', 'finetune_config']
    if any(key in card_data for key in finetune_keys):
        return 'finetune'

    return 'other'


def _classify_by_name_fallback(model_name: str) -> str:
    """
    回退方案：基于模型名称进行分类（当没有标签信息时）

    Args:
        model_name: 模型名称

    Returns:
        str: 模型类型
    """
    model_name_lower = model_name.lower()

    # Quantized 相关关键词（优先级最高）
    quantized_keywords = [
        # 格式标识
        '-gguf', '.gguf', 'gguf', '-gptq', '-awq', '-exl2',
        # 量化位数 - 通用格式
        '-4bit', '-8bit', '-6bit', '-2bit',
        'int2', 'int4', 'int8',
        # Q系列量化
        '-q1_', '-q2_', '-q3_', '-q4_', '-q5_', '-q6_', '-q8_',
        'q1_', 'q2_', 'q3_', 'q4_', 'q5_', 'q6_', 'q8_',
        # 精度格式（仅保留 fp8，移除 bf16/fp16 以免误判）
        'fp8',
        # W/A量化格式
        'w4a8', 'w4a16', 'w2a8', 'w8a8', 'w4a4',
        # MLX格式
        'mlx-4bit', 'mlx-8bit', 'mlx-6bit',
        # 其他标识
        '-quantized', '_quantized', 'quantized'
    ]
    if any(keyword in model_name_lower for keyword in quantized_keywords):
        return 'quantized'

    # LoRA 相关关键词
    lora_keywords = ['lora', 'low-rank-adaptation', 'low-rank']
    if any(keyword in model_name_lower for keyword in lora_keywords):
        return 'lora'

    # Adapter 相关关键词
    adapter_keywords = ['adapter', 'adapters', 'peft', 'prefix-tuning', 'prompt-tuning']
    if any(keyword in model_name_lower for keyword in adapter_keywords):
        return 'adapter'

    # Merge 相关关键词
    merge_keywords = ['-merge', '_merge', '-merged', '_merged']
    if any(keyword in model_name_lower for keyword in merge_keywords):
        return 'merge'

    # Finetune 相关关键词
    finetune_keywords = [
        'finetune', 'fine-tune', 'fine-tuned', 'finetuned',
        'custom-trained', 'custom-trained-model', 'trained-on'
    ]
    if any(keyword in model_name_lower for keyword in finetune_keywords):
        return 'finetune'

    # 检查是否为官方原始模型
    official_patterns = ['baidu/', 'paddlepaddle/']
    if any(pattern in model_name_lower for pattern in official_patterns):
        return 'original'

    return 'other'


def get_model_tree_children(base_model_id: str, max_depth: int = 1) -> List[Dict]:
    """
    获取指定模型的直接衍生模型（通过 HuggingFace API 的 base_model filter）

    Args:
        base_model_id: 基础模型ID（如 'baidu/ERNIE-4.5-21B-A3B-PT'）
        max_depth: 搜索深度，默认为1（只获取直接衍生）

    Returns:
        List[Dict]: 衍生模型信息列表
    """
    try:
        # 验证基础模型存在
        try:
            base_info = model_info(base_model_id)
            print(f"📊 获取 {base_model_id} 的model tree...")
        except Exception as e:
            print(f"⚠️ 基础模型 {base_model_id} 不存在或无法访问: {e}")
            return []

        # 使用 HuggingFace 官方的 base_model filter 功能
        # 这是正确的 Model Tree 查找方法
        try:
            derivatives = list(list_models(
                filter=f"base_model:{base_model_id}",
                full=True,
                limit=1000  # 增加限制以获取所有衍生模型
            ))

            if not derivatives:
                print(f"  ⚪ 没有找到衍生模型")
                return []

            print(f"  ✅ 找到 {len(derivatives)} 个衍生模型")

            # 转换为标准格式
            related_models = []
            for deriv in derivatives:
                try:
                    # 获取详细信息（包括下载量） - 必须使用 expand 参数
                    deriv_info = model_info(deriv.id, expand=["downloadsAllTime"])

                    # 获取下载量 - 优先使用 downloads_all_time，回退到 downloads
                    downloads = getattr(deriv_info, 'downloads_all_time', None) or getattr(deriv_info, 'downloads', 0) or 0

                    model_data = {
                        'id': deriv.id,
                        'author': deriv.author or 'Unknown',
                        'tags': getattr(deriv, 'tags', []),  # 🔧 修复：从 deriv 获取 tags（deriv_info.tags 为 None）
                        'downloads': downloads,
                        'pipeline_tag': getattr(deriv, 'pipeline_tag', None),  # 🔧 修复：从 deriv 获取
                        'created_at': getattr(deriv, 'created_at', None),
                        'last_modified': getattr(deriv, 'last_modified', None),
                        'likes': getattr(deriv, 'likes', 0)
                    }
                    related_models.append(model_data)

                except Exception as e:
                    print(f"    ⚠️ 获取 {deriv.id} 详情失败: {e}")
                    continue

            print(f"  ✅ 成功处理 {len(related_models)} 个衍生模型")
            return related_models

        except Exception as e:
            print(f"  ❌ 通过 base_model filter 查找失败: {e}")
            return []

    except Exception as e:
        print(f"❌ 获取 {base_model_id} 的model tree失败: {e}")
        return []


def extract_related_models_from_card(card_data: dict, base_model_id: str) -> List[str]:
    """
    从模型card中提取相关模型ID
    """
    related_models = []

    if not card_data:
        return related_models

    # 递归查找所有文本内容
    def extract_text(obj):
        texts = []
        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                texts.extend(extract_text(value))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(extract_text(item))
        return texts

    all_text = extract_text(card_data)
    combined_text = ' '.join(all_text).lower()

    # 查找相关的模型引用
    base_name = base_model_id.split('/')[-1].lower()

    # 查找模式
    patterns = [
        rf'{base_model_id.lower()}',
        rf'{base_name}',
        'based on',
        'finetuned from',
        'adapter for',
        'lora for'
    ]

    # 从文本中提取模型ID
    import re
    for text in all_text:
        # 查找模型ID模式
        model_id_pattern = r'([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)'
        matches = re.findall(model_id_pattern, text)

        for match in matches:
            if match != base_model_id and match not in related_models:
                # 验证是否是有效的模型
                try:
                    model_info(match)  # 验证模型存在
                    related_models.append(match)
                except:
                    continue

    return related_models


def is_genuine_derivative(model_info, base_model_id: str) -> bool:
    """
    验证一个模型是否真的是基础模型的衍生版本

    Args:
        model_info: HuggingFace模型信息对象
        base_model_id: 基础模型ID

    Returns:
        bool: 是否为真实的衍生模型
    """
    try:
        # 检查模型card内容
        if hasattr(model_info, 'card_data') and model_info.card_data:
            card_content = str(model_info.card_data).lower()
            base_id_lower = base_model_id.lower()
            base_name = base_model_id.split('/')[-1].lower()

            # 明确的衍生指标
            derivative_indicators = [
                f'based on {base_id_lower}',
                f'finetuned from {base_id_lower}',
                f'trained on {base_id_lower}',
                f'adapter for {base_id_lower}',
                f'lora adapter for {base_id_lower}',
                f'{base_name} finetune',
                f'{base_name} adapter',
                f'{base_name} lora'
            ]

            # 检查是否包含衍生指标
            if any(indicator in card_content for indicator in derivative_indicators):
                return True

        # 检查模型名称模式
        model_name = model_info.modelId.lower()
        base_name = base_model_id.split('/')[-1].lower()

        # 名称模式检查
        derivative_patterns = [
            f'{base_name}-finetune',
            f'{base_name}-adapter',
            f'{base_name}-lora',
            f'{base_name}-fine-tuned',
            f'{base_name}-adapted',
            f'finetuned-{base_name}',
            f'adapter-{base_name}',
            f'lora-{base_name}'
        ]

        if any(pattern in model_name for pattern in derivative_patterns):
            return True

        return False

    except Exception:
        return False


def is_derivative_model(model, base_model_id: str) -> bool:
    """
    判断一个模型是否是基础模型的衍生模型

    Args:
        model: HuggingFace模型对象
        base_model_id: 基础模型ID

    Returns:
        bool: 是否为衍生模型
    """
    model_id = model.id.lower()
    base_name = base_model_id.split('/')[-1].lower()

    # 检查模型名关系
    derivative_patterns = [
        f"{base_name}-",
        f"-{base_name}",
        f"finetuned-{base_name}",
        f"{base_name}-finetune",
        f"adapter-{base_name}",
        f"{base_name}-adapter",
        f"lora-{base_name}",
        f"{base_name}-lora"
    ]

    # 检查标签
    derivative_tags = ['fine-tuned', 'adapter', 'lora', 'peft']

    # 名字匹配或标签匹配
    name_match = any(pattern in model_id for pattern in derivative_patterns)
    tag_match = (hasattr(model, 'tags') and
                any(tag in [t.lower() for t in model.tags] for tag in derivative_tags))

    return name_match or tag_match


def get_all_ernie_derivatives(include_paddleocr: bool = True) -> Tuple[pd.DataFrame, int]:
    """
    获取所有 ERNIE-4.5 和 PaddleOCR-VL 相关模型，结合：
    - 全局搜索（ERNIE-4.5 / PaddleOCR-VL）
    - 官方账号模型（baidu / PaddlePaddle）
    - Model Tree 衍生
    - 以 base_model 为关键词的补充搜索

    与脚本版对齐：标记 official 为 original，Model Tree 命中与 search 合并为 both，
    保留 base_model_from_api，并带入更多字段（likes/library/pipeline/时间戳）。
    """
    print("🚀 开始获取ERNIE-4.5和PaddleOCR-VL模型...")

    all_models: List[Dict] = []
    processed_ids: Set[str] = set()
    official_models: Dict[str, Dict] = {}

    search_terms = ['ERNIE-4.5', 'PaddleOCR-VL']

    # ---------- 辅助函数 ----------
    def normalize_tags(tags):
        return tags if isinstance(tags, list) else (tags if tags is not None else [])

    def parse_base_from_card(card_data):
        if not card_data:
            return None
        base_val = card_data.get('base_model') if isinstance(card_data, dict) else None
        if isinstance(base_val, list) and base_val:
            return base_val[0]
        if isinstance(base_val, str) and base_val:
            return base_val
        return None

    def _get_field(obj, name):
        """兼容 dict 和 huggingface_hub 返回对象的取值"""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def fetch_model_detail(model_id, model_obj=None):
        try:
            info = model_info(model_id, expand=["downloadsAllTime"])
        except Exception as e:
            print(f"  ⚠️ 获取 {model_id} 详情失败: {e}")
            return None

        card_data = None
        if hasattr(info, 'cardData') and info.cardData:
            if isinstance(info.cardData, dict):
                card_data = info.cardData
            elif hasattr(info.cardData, '__dict__'):
                card_data = info.cardData.__dict__

        tags = normalize_tags(_get_field(model_obj, 'tags') or getattr(info, 'tags', None))
        pipeline_tag = _get_field(model_obj, 'pipeline_tag') or getattr(info, 'pipeline_tag', None)
        # Hugging Face 某些新模型的 author 字段可能为空，回退到 repo owner（ID 前缀）
        publisher = (
            getattr(info, 'author', None)
            or _get_field(model_obj, 'author')
            or (model_id.split('/')[0] if '/' in model_id else 'Unknown')
        )
        downloads = (
            getattr(info, 'downloads_all_time', None)
            or getattr(info, 'downloads', 0)
            or _get_field(model_obj, 'downloads')
            or 0
        )

        base_from_api = parse_base_from_card(card_data)

        # 如果 cardData 中没有 base_model，尝试从 tags 中提取
        if not base_from_api and tags:
            for tag in tags:
                if isinstance(tag, str) and tag.startswith('base_model:'):
                    # 提取 base_model，格式如: base_model:PaddlePaddle/PaddleOCR-VL
                    # 或 base_model:adapter:PaddlePaddle/PaddleOCR-VL
                    parts = tag.split(':', 2)  # 最多分割成3部分
                    if len(parts) >= 2:
                        # base_model:ModelID 或 base_model:type:ModelID
                        candidate = parts[-1]  # 取最后一部分作为 model ID
                        # 验证是否是有效的 model ID 格式 (包含 /)
                        if '/' in candidate and not candidate.startswith('license:'):
                            base_from_api = candidate
                            break

        model_category = classify_model(model_id, publisher, base_from_api)
        model_type = classify_model_type(model_id, tags, pipeline_tag, card_data)

        return {
            'model_id': model_id,
            'publisher': publisher,
            'downloads': downloads,
            'tags': tags,
            'pipeline_tag': pipeline_tag,
            'likes': getattr(info, 'likes', None),
            'library_name': getattr(info, 'library_name', None),
            'created_at': getattr(info, 'created_at', None),
            'last_modified': getattr(info, 'last_modified', None),
            'card_data': card_data,
            'model_category': model_category,
            'model_type': model_type,
            'base_model_from_api': base_from_api,
        }

    def search_models_with_keyword(keyword: str, exclude_ids: Set[str]) -> List:
        try:
            results = list(list_models(
                search=keyword,
                full=True,
                limit=600,
                sort="downloads",
                direction=-1
            ))
            filtered = [m for m in results if m.id not in exclude_ids]
            print(f"  🔍 搜索 '{keyword}'：{len(results)} 条，去重后 {len(filtered)} 条")
            return filtered
        except Exception as e:
            print(f"  ⚠️ 搜索 '{keyword}' 失败: {e}")
            return []

    allowed_categories = {'ernie-4.5', 'paddleocr-vl'}

    def add_record(detail: Dict, data_source: str, base_model: str = None, is_original: bool = False):
        if detail is None:
            return
        model_id = detail['model_id']
        model_name = model_id.split('/')[-1]
        publisher = detail['publisher']
        base_model_val = base_model or detail.get('base_model_from_api')
        model_category = detail['model_category']

        # 仅保留目标系列
        if model_category not in allowed_categories:
            return
        if not include_paddleocr and model_category == 'paddleocr-vl':
            return

        record = {
            'date': date.today().isoformat(),
            'repo': 'Hugging Face',
            'model_name': model_name,
            'publisher': publisher,
            'download_count': detail['downloads'],
            'model_category': model_category,
            'model_type': detail['model_type'] if not is_original else 'original',
            'is_derivative': bool(base_model_val),
            'base_model': base_model_val,
            'data_source': data_source,
            'tags': detail['tags'],
            'likes': detail.get('likes'),
            'library_name': detail.get('library_name'),
            'pipeline_tag': detail.get('pipeline_tag'),
            'created_at': detail.get('created_at'),
            'last_modified': detail.get('last_modified'),
            'fetched_at': date.today().isoformat(),
            'base_model_from_api': detail.get('base_model_from_api')
        }
        all_models.append(record)
        processed_ids.add(model_id)

    # ---------- 1. 全局搜索 ----------
    print(f"\n🔍 全局搜索（{', '.join(search_terms)}）...")
    all_search_models = []
    for search_term in search_terms:
        all_search_models.extend(search_models_with_keyword(search_term, exclude_ids=set()))

    unique_search = {}
    for m in all_search_models:
        if m.id not in unique_search:
            unique_search[m.id] = m
    print(f"🔍 去重后共 {len(unique_search)} 条搜索结果")

    for model in unique_search.values():
        detail = fetch_model_detail(model.id, model)
        if detail is None:
            continue

        add_record(detail, data_source='search')

        if model.author in ['baidu', 'PaddlePaddle']:
            official_models[model.id] = {
                'id': model.id,
                'category': detail['model_category']
            }

    # ---------- 2. 补充官方模型列表 ----------
    print("\n🌳 扩充官方模型列表...")
    try:
        baidu_official = list(list_models(author="baidu", search="ERNIE-4.5", limit=150))
        paddle_official = list(list_models(author="PaddlePaddle", search="PaddleOCR-VL", limit=50))
        print(f"  baidu 账号官方模型 {len(baidu_official)} 个；PaddlePaddle {len(paddle_official)} 个")
        for m in baidu_official + paddle_official:
            cat = 'paddleocr-vl' if 'paddleocr-vl' in m.id.lower() else 'ernie-4.5'
            if not include_paddleocr and cat == 'paddleocr-vl':
                continue
            if cat not in allowed_categories:
                continue
            official_models.setdefault(m.id, {'id': m.id, 'category': cat})
            if m.id in processed_ids:
                # 已在搜索结果中，更新为 official
                for rec in all_models:
                    if f"{rec['publisher']}/{rec['model_name']}" == m.id:
                        rec['data_source'] = 'original'
                        rec['base_model'] = None
                        rec['is_derivative'] = False
                        rec['model_type'] = 'original'
                        break
            else:
                detail = fetch_model_detail(m.id, m)
                if detail:
                    add_record(detail, data_source='original', base_model=None, is_original=True)
    except Exception as e:
        print(f"  ⚠️ 获取官方模型列表失败: {e}")

    official_list = list(official_models.values())
    print(f"🌳 官方基座数量: {len(official_list)}")

    # ---------- 3. 为每个官方模型查 Model Tree + 补充关键词搜索 ----------
    for official in official_list:
        model_id = official['id']
        model_category = official['category']
        print(f"\n🌳 处理基座: {model_id} ({model_category})")

        # Model Tree
        derivatives = get_model_tree_children(model_id, max_depth=1)
        if derivatives:
            for deriv in derivatives:
                deriv_detail = fetch_model_detail(deriv['id'], deriv)
                if deriv_detail is None:
                    continue

                if deriv['id'] not in processed_ids:
                    # 🔧 修复：使用 Model Tree 提供的 base_model 重新分类
                    # 因为 fetch_model_detail 中的分类可能使用了错误的 base_from_api（可能为空）
                    deriv_detail['model_category'] = classify_model(
                        deriv['id'],
                        deriv_detail['publisher'],
                        model_id  # 使用 Model Tree 的 base_model，而不是 base_from_api
                    )
                    add_record(deriv_detail, data_source='model_tree', base_model=model_id)
                else:
                    # 更新已有记录为 both，补 base_model
                    for existing in all_models:
                        if f"{existing['publisher']}/{existing['model_name']}" == deriv['id']:
                            existing['data_source'] = 'both'
                            existing['base_model'] = existing.get('base_model') or model_id
                            existing['is_derivative'] = True
                            # 🔧 修复：也要重新分类已有记录
                            existing['model_category'] = classify_model(
                                deriv['id'],
                                existing['publisher'],
                                model_id
                            )
                            break

        # 关键词补充搜索（按基座名）
        base_keyword = model_id.split('/')[-1]
        extra_results = search_models_with_keyword(base_keyword, exclude_ids=processed_ids)
        for model in extra_results:
            detail = fetch_model_detail(model.id, model)
            if detail is None:
                continue
            # 🔧 修复：使用当前基座的 model_id 重新分类
            # 因为这是通过基座名称搜索到的，应该与当前基座相关
            detail['model_category'] = classify_model(
                model.id,
                detail['publisher'],
                model_id  # 使用当前基座的 model_id
            )
            if not include_paddleocr and detail['model_category'] == 'paddleocr-vl':
                continue
            # 强制认为与当前 base 相关（兜底补充）
            if model.id not in processed_ids:
                add_record(detail, data_source='search', base_model=model_id)
                # 若已有 base_model_from_api，保留
                if detail.get('base_model_from_api') and not detail.get('base_model'):
                    for rec in all_models:
                        if rec['model_name'] == model.id.split('/')[-1] and rec['publisher'] == detail['publisher']:
                            rec['base_model'] = detail['base_model_from_api']
                            break

    # ---------- 4. 转 DataFrame ----------
    df = pd.DataFrame(all_models)
    if not df.empty:
        if 'tags' in df.columns:
            df['tags'] = df['tags'].apply(lambda x: str(x) if isinstance(x, list) else (x if pd.notna(x) else '[]'))
        # 把时间字段转为字符串
        for col in ['created_at', 'last_modified', 'fetched_at']:
            if col in df.columns:
                df[col] = df[col].astype(str)

        print(f"\n📊 总计获取 {len(df)} 个模型，其中基座 {len(official_list)} 个")

    return df, len(all_models)


def update_ernie_model_tree(save_to_db: bool = True) -> Tuple[pd.DataFrame, int]:
    """
    更新ERNIE模型树数据（包含所有衍生模型）

    Args:
        save_to_db: 是否保存到数据库

    Returns:
        Tuple[DataFrame, int]: (更新的数据, 总数量)
    """
    print("🔄 开始更新ERNIE模型树数据...")

    # 获取所有ERNIE相关模型
    df, total_count = get_all_ernie_derivatives(include_paddleocr=True)

    if df.empty:
        print("⚠️ 没有获取到任何模型数据")
        return df, 0

    # 准备数据库格式（包含模型类型和标签信息）
    required_columns = ['date', 'repo', 'model_name', 'publisher', 'download_count']
    optional_columns = [
        'model_type',
        'model_category',
        'tags',
        'base_model',
        'data_source',
        'likes',
        'library_name',
        'pipeline_tag',
        'created_at',
        'last_modified',
        'fetched_at',
        'base_model_from_api',
    ]

    # 确保所有必需列都存在
    available_columns = [col for col in required_columns if col in df.columns]
    db_df = df[available_columns].copy()

    # 添加可选列（如果存在）
    for col in optional_columns:
        if col in df.columns:
            db_df[col] = df[col]
        else:
            db_df[col] = None  # 为缺失的列填充默认值

    # 将tags列表转换为字符串存储
    if 'tags' in db_df.columns:
        db_df['tags'] = db_df['tags'].apply(lambda x: str(x) if isinstance(x, list) else (x if pd.notna(x) else '[]'))

    # 保存到数据库
    if save_to_db:
        save_to_db(db_df, DB_PATH)
        print(f"💾 已保存 {len(db_df)} 条记录到数据库")

    return df, total_count


def get_new_derivatives_since(last_date: str) -> pd.DataFrame:
    """
    获取自指定日期以来的新增衍生模型

    Args:
        last_date: 上次更新日期 (YYYY-MM-DD)

    Returns:
        DataFrame: 新增的衍生模型
    """
    try:
        # 从数据库获取指定日期以来的数据
        from ..db import load_data_from_db
        recent_data = load_data_from_db(date_filter=last_date)

        if recent_data.empty:
            return pd.DataFrame()

        # 筛选ERNIE相关的衍生模型
        ernie_data = recent_data[
            recent_data['model_name'].str.contains('ernie|ERNIE|paddleocr|PaddleOCR', case=False, na=False)
        ].copy()

        return ernie_data

    except Exception as e:
        print(f"获取新增衍生模型失败: {e}")
        return pd.DataFrame()


def get_weekly_new_finetune_adapters(current_date: str, previous_date: str, model_series: str = 'ERNIE-4.5') -> Dict:
    """
    获取本周新增的Finetune和Adapter模型（用于周报展示）

    Args:
        current_date: 当前日期 (YYYY-MM-DD)
        previous_date: 对比日期 (YYYY-MM-DD)
        model_series: 模型系列 ('ERNIE-4.5' 或 'PaddleOCR-VL')

    Returns:
        Dict: 包含本周新增Finetune和Adapter模型信息的字典
    """
    try:
        from ..db import load_data_from_db

        # 获取两个日期的数据
        current_data = load_data_from_db(date_filter=current_date)
        previous_data = load_data_from_db(date_filter=previous_date)

        if current_data.empty:
            return {
                'new_finetune_models': [],
                'new_adapter_models': [],
                'new_lora_models': [],
                'total_new': 0,
                'summary': '本周没有新增模型数据'
            }

        # 🔧 修复：使用 model_category 字段精确筛选，而不是搜索 model_name
        # 根据 model_series 确定要筛选的 model_category
        if model_series == 'ERNIE-4.5':
            target_category = 'ernie-4.5'
        elif model_series == 'PaddleOCR-VL':
            target_category = 'paddleocr-vl'
        else:
            # 默认为 ERNIE-4.5
            target_category = 'ernie-4.5'

        # 筛选Hugging Face平台的指定系列模型（使用 model_category 字段）
        hf_current = current_data[
            (current_data['repo'] == 'Hugging Face') &
            (current_data['model_category'] == target_category)
        ].copy()

        if previous_data.empty:
            # 如果没有对比数据，假设所有都是新增的
            hf_previous = pd.DataFrame()
        else:
            hf_previous = previous_data[
                (previous_data['repo'] == 'Hugging Face') &
                (previous_data['model_category'] == target_category)
            ].copy()

        # 找出新增的模型（在当前数据中但不在对比数据中）
        if hf_previous.empty:
            new_models = hf_current.copy()
        else:
            previous_model_names = set(hf_previous['model_name'].tolist())
            new_models = hf_current[~hf_current['model_name'].isin(previous_model_names)].copy()

        if new_models.empty:
            return {
                'new_finetune_models': [],
                'new_adapter_models': [],
                'new_lora_models': [],
                'total_new': 0,
                'summary': '本周没有新增Finetune或Adapter模型'
            }

        # 🔧 修复：直接使用数据库中已经存储的 model_type 字段，而不是重新分类
        # 数据在入库时已经通过 classify_model_type() 正确分类了
        # 如果 model_type 列不存在或为空，才进行分类（兼容旧数据）
        if 'model_type' not in new_models.columns or new_models['model_type'].isna().all():
            print("⚠️ 警告：model_type 字段不存在或全部为空，尝试重新分类")
            new_models['model_type'] = new_models.apply(
                lambda row: classify_model_type(
                    row['model_name'],
                    eval(row['tags']) if pd.notna(row.get('tags')) and row.get('tags') else [],
                    None
                ),
                axis=1
            )

        # 按类型分类
        new_finetune = new_models[new_models['model_type'] == 'finetune'].copy()
        new_adapter = new_models[new_models['model_type'] == 'adapter'].copy()
        new_lora = new_models[new_models['model_type'] == 'lora'].copy()

        # 格式化输出
        def format_models(df):
            if df.empty:
                return []
            return df[['model_name', 'publisher', 'download_count']].to_dict('records')

        result = {
            'new_finetune_models': format_models(new_finetune),
            'new_adapter_models': format_models(new_adapter),
            'new_lora_models': format_models(new_lora),
            'total_new': len(new_models),
            'summary': f'本周共发现 {len(new_models)} 个新增模型，其中 Finetune {len(new_finetune)} 个，Adapter {len(new_adapter)} 个，LoRA {len(new_lora)} 个'
        }

        return result

    except Exception as e:
        print(f"获取本周新增Finetune/Adapter模型失败: {e}")
        return {
            'new_finetune_models': [],
            'new_adapter_models': [],
            'new_lora_models': [],
            'total_new': 0,
            'summary': f'获取数据时出错: {e}'
        }


def get_weekly_new_model_tree_derivatives(current_date: str, previous_date: str, model_series: str = 'ERNIE-4.5') -> Dict:
    """
    获取本周新增的 Model Tree 衍生模型（专门统计）

    注意：只统计通过 Model Tree 找到的衍生模型（base_model 字段不为空）

    Args:
        current_date: 当前日期 (YYYY-MM-DD)
        previous_date: 对比日期 (YYYY-MM-DD)
        model_series: 模型系列 ('ERNIE-4.5' 或 'PaddleOCR-VL')

    Returns:
        Dict: 包含本周新增Model Tree衍生模型信息的字典
    """
    try:
        from ..db import load_data_from_db

        # 获取两个日期的数据
        current_data = load_data_from_db(date_filter=current_date)
        previous_data = load_data_from_db(date_filter=previous_date)

        if current_data.empty:
            return {
                'new_model_tree_models': [],
                'total_new': 0,
                'summary': '本周没有新增模型数据'
            }

        # 🔧 修复：使用 model_category 字段精确筛选，而不是搜索 model_name
        # 根据 model_series 确定要筛选的 model_category
        if model_series == 'ERNIE-4.5':
            target_category = 'ernie-4.5'
        elif model_series == 'PaddleOCR-VL':
            target_category = 'paddleocr-vl'
        else:
            # 默认为 ERNIE-4.5
            target_category = 'ernie-4.5'

        # 只筛选 Hugging Face 平台的指定系列模型，且 base_model 不为空（Model Tree 衍生模型）
        hf_current = current_data[
            (current_data['repo'] == 'Hugging Face') &
            (current_data['model_category'] == target_category) &
            (current_data['base_model'].notna()) &  # 只要 Model Tree 找到的
            (current_data['base_model'] != '') &    # base_model 不为空
            (current_data['base_model'] != 'None')  # 排除字符串 'None'
        ].copy()

        if previous_data.empty:
            # 如果没有对比数据，假设所有都是新增的
            hf_previous = pd.DataFrame()
        else:
            hf_previous = previous_data[
                (previous_data['repo'] == 'Hugging Face') &
                (previous_data['model_category'] == target_category) &
                (previous_data['base_model'].notna()) &
                (previous_data['base_model'] != '') &
                (previous_data['base_model'] != 'None')
            ].copy()

        # 找出新增的模型（在当前数据中但不在对比数据中）
        if hf_previous.empty:
            new_models = hf_current.copy()
        else:
            previous_model_names = set(hf_previous['model_name'].tolist())
            new_models = hf_current[~hf_current['model_name'].isin(previous_model_names)].copy()

        if new_models.empty:
            return {
                'new_model_tree_models': [],
                'total_new': 0,
                'summary': '本周没有新增 Model Tree 衍生模型'
            }

        # 🔧 修复：直接使用数据库中已经存储的 model_type 字段，而不是重新分类
        # 数据在入库时已经通过 classify_model_type() 正确分类了
        # 如果 model_type 列不存在或为空，才进行分类（兼容旧数据）
        if 'model_type' not in new_models.columns or new_models['model_type'].isna().all():
            print("⚠️ 警告：model_type 字段不存在或全部为空，尝试重新分类")
            new_models['model_type'] = new_models.apply(
                lambda row: classify_model_type(
                    row['model_name'],
                    eval(row['tags']) if pd.notna(row.get('tags')) and row.get('tags') else [],
                    None
                ),
                axis=1
            )

        # 格式化输出（增加 base_model 和 model_type 信息）
        def format_models(df):
            if df.empty:
                return []
            # 增加 base_model 和 model_type 列，方便在周报中显示详细信息
            if 'base_model' in df.columns and 'model_type' in df.columns:
                return df[['model_name', 'publisher', 'download_count', 'base_model', 'model_type']].to_dict('records')
            else:
                return df[['model_name', 'publisher', 'download_count']].to_dict('records')

        result = {
            'new_model_tree_models': format_models(new_models),
            'total_new': len(new_models),
            'summary': f'本周 Model Tree 新增 {len(new_models)} 个衍生模型'
        }

        return result

    except Exception as e:
        print(f"获取本周新增 Model Tree 衍生模型失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'new_model_tree_models': [],
            'total_new': 0,
            'summary': f'获取数据时出错: {e}'
        }


if __name__ == "__main__":
    # 测试功能
    print("=== 测试ERNIE模型树获取功能 ===")

    # 测试分类功能
    test_cases = [
        ("ernie-4.5-8b", "baidu"),
        ("ernie-4.5-8b-finetuned", "user123"),
        ("paddleocr-vl", "PaddlePaddle"),
        ("ernie-3.0", "baidu"),
        ("some-other-model", "user")
    ]

    print("\n🧪 测试模型分类:")
    for model_name, publisher in test_cases:
        category = classify_model(model_name, publisher)
        print(f"  {model_name} -> {category}")

    # 测试获取模型树
    print("\n🌳 测试获取模型树:")
    df, count = get_all_ernie_derivatives(include_paddleocr=True)
    print(f"总共获取到 {count} 个模型")

    if not df.empty:
        print("\n前5个模型:")
        print(df[['model_name', 'publisher', 'download_count', 'model_category']].head())
