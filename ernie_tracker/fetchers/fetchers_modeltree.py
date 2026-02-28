"""
Model Tree 功能模块 - 获取官方模型的衍生模型
支持获取 Finetune 和 Adapter 模型，并智能分类
"""
from huggingface_hub import list_models, model_info
from datetime import date, datetime
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
        if not name:
            return False
        n = str(name).lower()
        compact = re.sub(r'[^a-z0-9]+', '', n)
        return ('paddleocr' in n and 'vl' in n) or ('paddleocr' in compact and 'vl' in compact)

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
                    # 第一次调用：不带expand，获取created_at等基础字段
                    deriv_basic = model_info(deriv.id)

                    # 第二次调用：带expand，获取downloadsAllTime
                    deriv_info = model_info(deriv.id, expand=["downloadsAllTime"])

                    # 将created_at从basic对象复制到expand对象
                    if hasattr(deriv_basic, 'created_at') and not getattr(deriv_info, 'created_at', None):
                        deriv_info.created_at = deriv_basic.created_at
                    if hasattr(deriv_basic, 'last_modified') and not getattr(deriv_info, 'last_modified', None):
                        deriv_info.last_modified = deriv_basic.last_modified

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
            # 第一次调用：不带expand，获取created_at等基础字段
            info_basic = model_info(model_id)

            # 第二次调用：带expand，获取downloadsAllTime
            info = model_info(model_id, expand=["downloadsAllTime"])

            # 将created_at从basic对象复制到expand对象
            if hasattr(info_basic, 'created_at') and not getattr(info, 'created_at', None):
                info.created_at = info_basic.created_at
            if hasattr(info_basic, 'last_modified') and not getattr(info, 'last_modified', None):
                info.last_modified = info_basic.last_modified

            # 调试：检查created_at是否成功复制
            if model_id == 'baidu/ERNIE-4.5-0.3B-PT':
                print(f"  调试 {model_id}:")
                print(f"    - created_at (after copy): {getattr(info, 'created_at', None)}")
                print(f"    - downloads_all_time: {getattr(info, 'downloads_all_time', None)}")
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
            'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'base_model_from_api': detail.get('base_model_from_api'),
            'url': f"https://huggingface.co/{model_id}"  # 模型详情页URL
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
        'url',  # 🔧 修复：之前遗漏了 url 字段
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


# =============================================================================
# ModelScope Model Tree 功能模块
# =============================================================================

def get_modelscope_model_tree_children(base_model_id: str, driver=None, progress_callback=None) -> List[Dict]:
    """
    获取 ModelScope 模型的衍生模型（通过解析页面 HTML）

    Args:
        base_model_id: 基础模型ID（如 'PaddlePaddle/PaddleOCR-VL'）
        driver: Selenium WebDriver 实例（可选，如果不提供则创建新的）
        progress_callback: 进度回调函数

    Returns:
        List[Dict]: 衍生模型信息列表
    """
    from ..utils import create_chrome_driver
    from ..config import SELENIUM_TIMEOUT
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from modelscope.hub.api import HubApi
    import time
    import re

    print(f"\n📊 获取 {base_model_id} 的 ModelScope Model Tree...")

    should_close_driver = False
    if driver is None:
        driver = create_chrome_driver()
        should_close_driver = True

    try:
        # 构建模型页面URL
        model_url = f"https://modelscope.cn/models/{base_model_id}"
        print(f"  访问: {model_url}")
        driver.get(model_url)

        # 等待页面加载
        try:
            WebDriverWait(driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)  # 额外等待确保动态内容加载
        except TimeoutException:
            print(f"  ⚠️ 页面加载超时")
            return []

        # 查找所有模型血缘（genealogy）相关的衍生类型元素
        # 直接查找 span.antd5-tree-node-content-wrapper（这是真正可点击的元素）
        try:
            # 查找所有可点击的 tree node wrapper 元素
            node_wrappers = driver.find_elements(
                By.CSS_SELECTOR,
                "span.antd5-tree-node-content-wrapper"
            )

            if not node_wrappers:
                print(f"  ⚪️ 没有找到衍生类型")
                return []

            print(f"  ✅ 找到 {len(node_wrappers)} 个 node wrapper 元素")

            # 过滤出真正的衍生类型（排除"当前模型"）
            derivative_types = []
            for wrapper in node_wrappers:
                try:
                    # 在每个 wrapper 内部查找 span.antd5-tree-title，然后再找 div.acss-1lekzkb
                    try:
                        tree_title = wrapper.find_element(By.CSS_SELECTOR, "span.antd5-tree-title")
                        content_div = tree_title.find_element(By.CSS_SELECTOR, "div.acss-1lekzkb")
                    except:
                        continue

                    # 获取元素文本，检查是否为"当前模型"
                    element_text = content_div.text.strip()

                    if not element_text:
                        continue

                    # 检查是否包含"当前模型"标记
                    if "当前模型" in element_text:
                        continue

                    # 提取中英文名称
                    # 根据HTML结构，应该是"微调 Finetunes"或类似格式
                    text_parts = element_text.split('\n')
                    if len(text_parts) >= 2:
                        name_zh = text_parts[0].strip()
                        name_en = text_parts[1].strip()

                        # 提取模型数量（通常在最后一个部分）
                        count_match = re.search(r'共(\d+)个模型', element_text)
                        count = int(count_match.group(1)) if count_match else 0

                        if count > 0:
                            # 🔧 关键修复：需要点击的是内部的 div.acss-hd4erf（包含中文标题的div）
                            # 而不是外层的 wrapper
                            try:
                                clickable_element = content_div.find_element(By.CSS_SELECTOR, "div.acss-hd4erf")
                            except:
                                # 如果找不到，回退到使用wrapper
                                clickable_element = wrapper

                            derivative_types.append({
                                'element': clickable_element,  # 使用内部的可点击div
                                'name_zh': name_zh,
                                'name_en': name_en,
                                'count': count
                            })
                            print(f"    📂 {name_zh} / {name_en}: {count}个模型")

                except Exception as e:
                    print(f"    ⚠️ 解析衍生类型元素时出错: {e}")
                    continue

            if not derivative_types:
                print(f"  ⚪️ 没有找到有效的衍生类型")
                return []

            # 初始化 ModelScope API
            api = HubApi()
            all_derivatives = []

            # 🔧 新策略：先打开侧边栏（只点击第一个衍生类型）
            # 然后在侧边栏内部通过点击标签切换不同类型
            print(f"\n  📂 打开侧边栏...")

            if not derivative_types:
                return []

            # 使用第一个衍生类型打开侧边栏
            first_type = derivative_types[0]
            first_element = first_type['element']

            try:
                # 滚动到元素可见
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", first_element)
                time.sleep(0.5)

                # 点击第一个衍生类型打开侧边栏
                first_element.click()
                print(f"    ✅ 已点击第一个衍生类型打开侧边栏")

                # 等待侧边栏加载
                print(f"    ⏳ 等待侧边栏加载...")
                before_click_links = len(driver.find_elements(By.CSS_SELECTOR, "a[href*='/models/']"))

                max_wait = 10
                waited = 0
                while waited < max_wait:
                    time.sleep(1)
                    waited += 1
                    current_links = len(driver.find_elements(By.CSS_SELECTOR, "a[href*='/models/']"))
                    if current_links > before_click_links:
                        print(f"    ✅ 侧边栏已加载（等待了 {waited} 秒）")
                        break
                else:
                    print(f"    ⚠️ 等待 {max_wait} 秒后侧边栏仍未加载")
                    return []

            except Exception as e:
                print(f"    ❌ 打开侧边栏失败: {e}")
                return []

            # 🔧 关键改进：在侧边栏内部通过点击标签切换不同类型
            # 查找侧边栏内的标签元素
            try:
                # 等待侧边栏完全加载
                time.sleep(2)

                # 查找所有衍生类型标签
                tab_elements = driver.find_elements(By.CSS_SELECTOR, "div.acss-xqwyei")

                if not tab_elements:
                    print(f"    ⚠️ 侧边栏中没有找到标签元素")
                    # 使用原来的逻辑（逐个点击外部元素）
                    print(f"    📋 回退到原来的点击方式...")
                else:
                    print(f"    ✅ 找到 {len(tab_elements)} 个侧边栏标签")

                    # 为每个标签建立映射：标签文本 -> 衍生类型信息
                    tab_mapping = []
                    for tab in tab_elements:
                        try:
                            tab_text = tab.text.strip()
                            # 提取中文名称（第一行）
                            name_zh = tab_text.split('\n')[0] if '\n' in tab_text else tab_text

                            # 在 derivative_types 中找到对应的类型信息
                            matching_type = None
                            for dt in derivative_types:
                                if dt['name_zh'] == name_zh:
                                    matching_type = dt
                                    break

                            if matching_type:
                                tab_mapping.append({
                                    'tab': tab,
                                    'name_zh': matching_type['name_zh'],
                                    'name_en': matching_type['name_en'],
                                    'count': matching_type['count']
                                })
                        except:
                            continue

                    print(f"    📋 建立了 {len(tab_mapping)} 个标签映射")

                    # 逐个点击标签并获取模型
                    for idx, tab_info in enumerate(tab_mapping):
                        try:
                            name_zh = tab_info['name_zh']
                            name_en = tab_info['name_en']
                            tab = tab_info['tab']

                            print(f"\n  [{idx + 1}/{len(tab_mapping)}] 切换到: {name_zh} / {name_en}")

                            # 点击标签
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", tab)
                            time.sleep(0.5)

                            # 使用JavaScript点击（更可靠）
                            driver.execute_script("arguments[0].click();", tab)
                            print(f"    ✅ 已切换标签")

                            # 等待内容加载
                            time.sleep(2)

                            # 查找当前标签下的模型卡片
                            all_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/models/']")
                            model_cards = []
                            for link in all_links:
                                href = link.get_attribute('href')
                                if href and '/models/' in href:
                                    if not any(x in href for x in ['/summary', '/files', '/feedback', '/file/view']):
                                        model_cards.append(link)

                            if not model_cards:
                                print(f"    ⚪️ 当前标签下没有找到模型卡片")
                                continue

                            print(f"    ✅ 找到 {len(model_cards)} 个模型卡片")

                            # 提取模型信息
                            for card in model_cards:
                                try:
                                    href = card.get_attribute('href')
                                    if not href or '/models/' not in href:
                                        continue

                                    model_id = href.split('/models/')[-1]
                                    if '?' in model_id:
                                        model_id = model_id.split('?')[0]

                                    print(f"      🔍 检查模型: {model_id}")

                                    # 跳过基础模型本身
                                    if model_id == base_model_id:
                                        print(f"        ⏭️ 跳过（这是基础模型本身）")
                                        continue

                                    print(f"      📦 {model_id}")

                                    # 使用API获取模型详细信息
                                    try:
                                        info = api.get_model(model_id, revision="master")
                                        downloads = info.get("Downloads", 0)

                                        from datetime import datetime
                                        created_at = None
                                        last_modified = None

                                        if "CreatedTime" in info and info["CreatedTime"]:
                                            try:
                                                created_at = datetime.fromtimestamp(info["CreatedTime"]).strftime('%Y-%m-%d')
                                            except:
                                                pass

                                        if "LastUpdatedTime" in info and info["LastUpdatedTime"]:
                                            try:
                                                last_modified = datetime.fromtimestamp(info["LastUpdatedTime"]).strftime('%Y-%m-%d')
                                            except:
                                                pass

                                        publisher = model_id.split('/')[0] if '/' in model_id else 'Unknown'

                                        derivative_info = {
                                            'id': model_id,
                                            'author': publisher,
                                            'downloads': downloads,
                                            'pipeline_tag': None,
                                            'tags': [],
                                            'created_at': created_at,
                                            'last_modified': last_modified,
                                            'likes': info.get('Likes', 0),
                                            'model_type': name_en.lower(),
                                            'base_model': base_model_id,
                                            'name_zh': name_zh,
                                            'name_en': name_en
                                        }

                                        all_derivatives.append(derivative_info)

                                    except Exception as e:
                                        print(f"        ⚠️ API获取失败: {e}")
                                        publisher = model_id.split('/')[0] if '/' in model_id else 'Unknown'
                                        derivative_info = {
                                            'id': model_id,
                                            'author': publisher,
                                            'downloads': 0,
                                            'pipeline_tag': None,
                                            'tags': [],
                                            'created_at': None,
                                            'last_modified': None,
                                            'likes': 0,
                                            'model_type': name_en.lower(),
                                            'base_model': base_model_id,
                                            'name_zh': name_zh,
                                            'name_en': name_en
                                        }
                                        all_derivatives.append(derivative_info)

                                except Exception as e:
                                    print(f"      ⚠️ 处理模型时出错: {e}")
                                    continue

                        except Exception as e:
                            print(f"    ⚠️ 处理标签时出错: {e}")
                            continue

                    print(f"\n  ✅ 总共获取 {len(all_derivatives)} 个衍生模型")
                    return all_derivatives

            except Exception as e:
                print(f"    ❌ 侧边栏标签切换失败: {e}")
                import traceback
                traceback.print_exc()
                # 继续执行，尝试使用原来的逻辑
                pass

            # 如果侧边栏标签切换失败，使用原来的逐个点击方式
            print(f"\n  📋 使用原来的逐个点击方式...")
            for idx, deriv_type in enumerate(derivative_types):
                try:
                    name_zh = deriv_type['name_zh']
                    name_en = deriv_type['name_en']
                    count = deriv_type['count']
                    element = deriv_type['element']

                    print(f"\n  [{idx + 1}/{len(derivative_types)}] 处理衍生类型: {name_zh} / {name_en}")

                    # 点击衍生类型元素，打开侧边栏
                    try:
                        # 滚动到元素可见
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                        time.sleep(0.5)

                        # 点击元素
                        element.click()
                        print(f"    ✅ 已点击衍生类型")

                        # 🔧 关键修复：等待侧边栏真的出现，而不是简单等待固定时间
                        # 等待链接数量增加（说明侧边栏已经加载了新内容）
                        print(f"    ⏳ 等待侧边栏加载...")

                        # 先获取当前链接数量
                        before_click_links = len(driver.find_elements(By.CSS_SELECTOR, "a[href*='/models/']"))

                        # 等待最多10秒，直到链接数量增加
                        max_wait = 10
                        waited = 0
                        while waited < max_wait:
                            time.sleep(1)
                            waited += 1
                            current_links = len(driver.find_elements(By.CSS_SELECTOR, "a[href*='/models/']"))
                            if current_links > before_click_links:
                                print(f"    ✅ 侧边栏已加载（等待了 {waited} 秒）")
                                break
                        else:
                            print(f"    ⚠️ 等待 {max_wait} 秒后侧边栏仍未加载新内容")

                    except Exception as e:
                        print(f"    ⚠️ 点击衍生类型失败: {e}")
                        continue

                    # 查找侧边栏中的模型卡片
                    # 根据HTML结构，模型卡片在侧边栏中，包含模型名称
                    try:
                        # 尝试多种选择器查找模型卡片
                        model_cards = []

                        # 方法1: 直接查找模型链接（排除基础模型本身的子页面）
                        all_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/models/']")
                        for link in all_links:
                            href = link.get_attribute('href')
                            if href and '/models/' in href:
                                # 排除基础模型本身的子页面（如 summary、files、feedback）
                                # 只保留真正的模型链接（格式：/models/username/modelname）
                                if not any(x in href for x in ['/summary', '/files', '/feedback', '/file/view']):
                                    model_cards.append(link)

                        if not model_cards:
                            print(f"    ⚪️ 侧边栏中没有找到模型卡片")
                            # 尝试关闭侧边栏（按ESC键或点击背景）
                            try:
                                from selenium.webdriver.common.keys import Keys
                                driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                                time.sleep(0.5)
                            except:
                                pass
                            continue

                        print(f"    ✅ 找到 {len(model_cards)} 个模型卡片")

                        # 提取模型信息
                        for card in model_cards:
                            try:
                                # 获取模型ID（从href属性）
                                href = card.get_attribute('href')
                                if not href or '/models/' not in href:
                                    print(f"      ⚠️ 跳过无效链接: href={href}")
                                    continue

                                model_id = href.split('/models/')[-1]
                                if '?' in model_id:
                                    model_id = model_id.split('?')[0]

                                print(f"      🔍 检查模型: {model_id}")

                                # 跳过基础模型本身
                                if model_id == base_model_id:
                                    print(f"        ⏭️ 跳过（这是基础模型本身）")
                                    continue

                                print(f"      📦 {model_id}")

                                # 使用API获取模型详细信息
                                try:
                                    info = api.get_model(model_id, revision="master")

                                    # 提取下载量
                                    downloads = info.get("Downloads", 0)

                                    # 提取时间字段
                                    from datetime import datetime
                                    created_at = None
                                    last_modified = None

                                    if "CreatedTime" in info and info["CreatedTime"]:
                                        try:
                                            created_at = datetime.fromtimestamp(info["CreatedTime"]).strftime('%Y-%m-%d')
                                        except:
                                            pass

                                    if "LastUpdatedTime" in info and info["LastUpdatedTime"]:
                                        try:
                                            last_modified = datetime.fromtimestamp(info["LastUpdatedTime"]).strftime('%Y-%m-%d')
                                        except:
                                            pass

                                    # 提取发布者
                                    publisher = model_id.split('/')[0] if '/' in model_id else 'Unknown'

                                    # 创建衍生模型记录
                                    derivative_info = {
                                        'id': model_id,
                                        'author': publisher,
                                        'downloads': downloads,
                                        'pipeline_tag': None,
                                        'tags': [],
                                        'created_at': created_at,
                                        'last_modified': last_modified,
                                        'likes': info.get('Likes', 0),
                                        'model_type': name_en.lower(),  # finetune, quantized, etc.
                                        'base_model': base_model_id,
                                        'name_zh': name_zh,
                                        'name_en': name_en
                                    }

                                    all_derivatives.append(derivative_info)

                                except Exception as e:
                                    print(f"        ⚠️ API获取失败: {e}")
                                    # 即使API失败，也可以保存基本信息
                                    publisher = model_id.split('/')[0] if '/' in model_id else 'Unknown'
                                    derivative_info = {
                                        'id': model_id,
                                        'author': publisher,
                                        'downloads': 0,
                                        'pipeline_tag': None,
                                        'tags': [],
                                        'created_at': None,
                                        'last_modified': None,
                                        'likes': 0,
                                        'model_type': name_en.lower(),
                                        'base_model': base_model_id,
                                        'name_zh': name_zh,
                                        'name_en': name_en
                                    }
                                    all_derivatives.append(derivative_info)

                            except Exception as e:
                                print(f"      ⚠️ 处理模型卡片时出错: {e}")
                                continue

                        # 关闭侧边栏（按ESC键或点击背景）
                        try:
                            from selenium.webdriver.common.keys import Keys
                            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                            time.sleep(0.5)
                        except:
                            pass

                        if progress_callback:
                            progress_callback(idx + 1, total=len(derivative_types))

                    except Exception as e:
                        print(f"    ⚠️ 处理侧边栏时出错: {e}")
                        continue

                except Exception as e:
                    print(f"  ⚠️ 处理衍生类型时出错: {e}")
                    continue

            print(f"\n  ✅ 总共获取 {len(all_derivatives)} 个衍生模型")
            return all_derivatives

        except NoSuchElementException:
            print(f"  ⚪️ 未找到模型血缘元素")
            return []

    except Exception as e:
        print(f"  ❌ 获取 {base_model_id} 的 ModelScope Model Tree 失败: {e}")
        import traceback
        traceback.print_exc()
        return []

    finally:
        if should_close_driver and driver:
            driver.quit()


def get_all_modelscope_derivatives(
    base_models: List[str] = None,
    auto_discover: bool = True,
    progress_callback=None
) -> Tuple[pd.DataFrame, int]:
    """
    获取 ModelScope 上所有指定基础模型的衍生模型

    Args:
        base_models: 基础模型ID列表（如果为None且auto_discover=True，则自动从数据库发现）
        auto_discover: 是否自动从数据库中发现所有ModelScope官方模型
        progress_callback: 进度回调函数

    Returns:
        Tuple[DataFrame, int]: (衍生模型数据, 总数量)
    """
    from ..utils import create_chrome_driver
    import sqlite3

    # 如果没有提供基础模型列表，自动从数据库发现
    if base_models is None and auto_discover:
        print(f"\n🔍 自动发现 ModelScope 官方模型...")
        try:
            conn = sqlite3.connect(DB_PATH)

            # 查询所有ModelScope平台的官方模型
            query = """
                SELECT DISTINCT publisher, model_name
                FROM model_downloads
                WHERE repo = 'ModelScope'
                AND (
                    publisher IN ('百度', 'baidu', 'Paddle', 'PaddlePaddle', 'yiyan', '一言')
                    OR publisher LIKE '%百度%'
                    OR publisher LIKE '%baidu%'
                    OR publisher LIKE '%Paddle%'
                )
                ORDER BY publisher, model_name
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                print(f"  ⚠️ 数据库中没有找到 ModelScope 官方模型")
                base_models = []
            else:
                # 构建模型ID列表
                base_models = [f"{row['publisher']}/{row['model_name']}" for _, row in df.iterrows()]
                print(f"  ✅ 发现 {len(base_models)} 个官方模型")

                # 显示前10个模型
                for i, model_id in enumerate(base_models[:10]):
                    print(f"    {i+1}. {model_id}")
                if len(base_models) > 10:
                    print(f"    ... 还有 {len(base_models) - 10} 个模型")

        except Exception as e:
            print(f"  ❌ 自动发现失败: {e}")
            base_models = []

    # 如果仍然没有基础模型，使用默认列表
    if not base_models:
        base_models = [
            'PaddlePaddle/PaddleOCR-VL',
        ]
        print(f"\n📋 使用默认基础模型列表")

    print(f"\n🚀 开始获取 ModelScope 衍生模型...")
    print(f"📋 基础模型列表: {len(base_models)} 个")
    print(f"   {', '.join(base_models[:5])}")
    if len(base_models) > 5:
        print(f"   ... 还有 {len(base_models) - 5} 个模型")

    all_models = []
    processed_ids = set()

    driver = create_chrome_driver()

    try:
        for idx, base_model in enumerate(base_models, start=1):
            print(f"\n{'=' * 80}")
            print(f"[{idx}/{len(base_models)}] 处理基础模型: {base_model}")
            print(f"{'=' * 80}")

            # 调用进度回调
            if progress_callback:
                progress_callback(idx)

            try:
                # 获取该基础模型的衍生模型
                derivatives = get_modelscope_model_tree_children(base_model, driver=driver)

                if derivatives:
                    print(f"  ✅ 获取到 {len(derivatives)} 个衍生模型")

                    for deriv in derivatives:
                        model_id = deriv['id']

                        # 跳过重复的模型
                        if model_id in processed_ids:
                            print(f"      ⏭️ 跳过重复模型: {model_id}")
                            continue

                        processed_ids.add(model_id)

                        # 创建记录
                        record = {
                            'date': date.today().isoformat(),
                            'repo': 'ModelScope',
                            'model_name': model_id.split('/')[-1] if '/' in model_id else model_id,
                            'publisher': deriv['author'],
                            'download_count': deriv['downloads'],
                            'model_category': classify_model(
                                deriv['id'],
                                deriv['author'],
                                deriv['base_model']
                            ),
                            'model_type': deriv.get('model_type', 'other'),
                            'base_model': deriv['base_model'],
                            'data_source': 'model_tree',
                            'tags': str(deriv.get('tags', [])),
                            'likes': deriv.get('likes'),
                            'library_name': None,
                            'pipeline_tag': deriv.get('pipeline_tag'),
                            'created_at': deriv.get('created_at'),
                            'last_modified': deriv.get('last_modified'),
                            'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'base_model_from_api': deriv['base_model'],
                            'search_keyword': deriv['base_model']
                        }

                        all_models.append(record)
                        print(f"    ✓ {deriv['name_zh']}: {model_id}")
                else:
                    print(f"  ⚪️ 没有找到衍生模型")

            except Exception as e:
                print(f"  ❌ 处理 {base_model} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    finally:
        driver.quit()

    # 转换为 DataFrame
    if all_models:
        df = pd.DataFrame(all_models)
        print(f"\n{'=' * 80}")
        print(f"✅ 成功获取 {len(df)} 个衍生模型")
        print(f"{'=' * 80}")
        return df, len(all_models)
    else:
        print(f"\n{'=' * 80}")
        print(f"⚠️ 没有找到任何衍生模型")
        print(f"{'=' * 80}")
        return pd.DataFrame(), 0


def update_modelscope_model_tree(
    save_to_db: bool = True,
    base_models: List[str] = None,
    auto_discover: bool = True,
    progress_callback=None
) -> Tuple[pd.DataFrame, int]:
    """
    更新 ModelScope Model Tree 数据（包含去重处理）

    Args:
        save_to_db: 是否保存到数据库
        base_models: 基础模型ID列表（如果为None且auto_discover=True，则自动从数据库发现）
        auto_discover: 是否自动从数据库中发现所有ModelScope官方模型
        progress_callback: 进度回调函数

    Returns:
        Tuple[DataFrame, int]: (更新的数据, 总数量)
    """
    print("\n🔄 开始更新 ModelScope Model Tree 数据...")

    # 获取衍生模型（自动发现所有官方模型）
    df, total_count = get_all_modelscope_derivatives(
        base_models=base_models,
        auto_discover=auto_discover,
        progress_callback=progress_callback
    )

    if df.empty:
        print("⚠️ 没有获取到任何衍生模型数据")
        return df, 0

    # 去重处理：检查数据库中是否已存在相同的模型
    if save_to_db:
        try:
            import sqlite3
            from ..db import load_data_from_db, save_to_db as save_to_db_func

            # 获取现有 ModelScope 数据
            conn = sqlite3.connect(DB_PATH)
            existing_query = """
                SELECT DISTINCT publisher, model_name
                FROM model_downloads
                WHERE repo = 'ModelScope'
            """
            existing_df = pd.read_sql_query(existing_query, conn)
            conn.close()

            if not existing_df.empty:
                # 创建已存在模型的集合
                existing_models = set(
                    f"{row['publisher']}/{row['model_name']}"
                    for _, row in existing_df.iterrows()
                )

                # 过滤掉已存在的模型
                df['model_key'] = df['publisher'] + '/' + df['model_name']
                new_df = df[~df['model_key'].isin(existing_models)].copy()
                new_df = new_df.drop(columns=['model_key'])

                print(f"📊 去重前: {len(df)} 条，去重后: {len(new_df)} 条")
                print(f"🗑️  过滤掉 {len(df) - len(new_df)} 条已存在的记录")

                if new_df.empty:
                    print("⚠️ 没有新的模型需要保存")
                    return df, 0

                df = new_df

            # 保存到数据库
            save_to_db_func(df, DB_PATH)
            print(f"💾 已保存 {len(df)} 条新记录到数据库")

        except Exception as e:
            print(f"❌ 保存数据时出错: {e}")
            import traceback
            traceback.print_exc()

    return df, total_count


if __name__ == "__main__":
    # 测试功能
    print("=== 测试 Model Tree 功能 ===")
    print("1. Hugging Face Model Tree")
    print("2. AI Studio Model Tree")
    print("3. ModelScope Model Tree (NEW)")
    print("4. 全部测试")
    print()

    choice = input("请选择测试模式 (1/2/3/4=全部, 默认=4): ").strip()

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

    # 测试Hugging Face Model Tree
    if choice in ['1', '4', '']:
        print("\n🌳 测试 Hugging Face Model Tree:")
        df, count = get_all_ernie_derivatives(include_paddleocr=True)
        print(f"总共获取到 {count} 个模型")

        if not df.empty:
            print("\n前5个模型:")
            print(df[['model_name', 'publisher', 'download_count', 'model_category']].head())

    # 测试AI Studio Model Tree
    if choice in ['2', '4', '']:
        print("\n🌳 测试 AI Studio Model Tree (测试模式):")
        df, count = update_aistudio_model_tree(save_to_db=False, test_mode=True)
        print(f"总共获取到 {count} 个衍生模型")

        if not df.empty:
            print("\n前5个衍生模型:")
            print(df[['model_name', 'publisher', 'download_count', 'model_type', 'base_model']].head())

    # 测试 ModelScope Model Tree
    if choice in ['3', '4', '']:
        print("\n🌳 测试 ModelScope Model Tree:")
        df, count = update_modelscope_model_tree(
            save_to_db=False,
            base_models=['PaddlePaddle/PaddleOCR-VL']
        )
        print(f"总共获取到 {count} 个衍生模型")

        if not df.empty:
            print("\n前5个衍生模型:")
            print(df[['model_name', 'publisher', 'download_count', 'model_type', 'base_model']].head())


# =============================================================================
# AI Studio Model Tree 功能模块
# =============================================================================

def get_aistudio_official_models():
    """
    从数据库获取所有AI Studio官方模型

    Returns:
        DataFrame: 官方模型数据，包含 model_name, publisher, url 等字段
    """
    try:
        from ..db import load_data_from_db
        import sqlite3
        import pandas as pd

        conn = sqlite3.connect(DB_PATH)

        # 获取AI Studio平台的数据
        query = """
            SELECT DISTINCT model_name, publisher, url
            FROM model_downloads
            WHERE repo = 'AI Studio'
            AND (
                publisher IN ('百度', 'baidu', 'Paddle', 'PaddlePaddle', 'yiyan', '一言')
                OR publisher LIKE '%百度%'
                OR publisher LIKE '%baidu%'
                OR publisher LIKE '%Paddle%'
            )
            AND url IS NOT NULL
            AND url != ''
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"📊 找到 {len(df)} 个AI Studio官方模型")
        return df

    except Exception as e:
        print(f"❌ 获取AI Studio官方模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_aistudio_model_tree(
    progress_callback=None,
    include_official_publishers=None,
    test_mode=False,
    save_to_db=False
):
    """
    获取AI Studio官方模型的Model Tree（衍生模型）

    Args:
        progress_callback: 进度回调函数，支持两种调用方式：
                          - progress_callback(processed_count) - 更新进度
                          - progress_callback(message) - 记录日志消息
        include_official_publishers: 官方发布者列表（默认使用标准列表）
        test_mode: 测试模式，只处理第一个模型
        save_to_db: 是否保存到数据库

    Returns:
        tuple: (DataFrame, total_count) 衍生模型数据和数量
    """

    def log(message):
        """记录日志，通过progress_callback传递给前端"""
        print(message)  # 保留控制台输出
        if progress_callback:
            # 尝试作为日志消息调用
            try:
                progress_callback(message)
            except:
                # 如果调用失败，说明callback可能只接受int参数（进度更新）
                pass
    from ..utils import create_chrome_driver
    from ..config import SELENIUM_TIMEOUT, DB_PATH
    from ..fetchers.selenium import AIStudioFetcher
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    import time
    import re
    import sqlite3

    log("\n" + "=" * 80)
    log("🌳 开始获取 AI Studio Model Tree")
    log("=" * 80)

    # 获取已存在的模型集合（用于跳过URL获取）
    existing_models_with_url = set()
    try:
        conn = sqlite3.connect(DB_PATH)
        existing_query = """
            SELECT DISTINCT publisher, model_name
            FROM model_downloads
            WHERE repo = 'AI Studio' AND url IS NOT NULL AND url != ''
        """
        existing_df = pd.read_sql_query(existing_query, conn)
        conn.close()

        if not existing_df.empty:
            existing_models_with_url = set(
                f"{row['publisher']}/{row['model_name']}"
                for _, row in existing_df.iterrows()
            )
            log(f"📚 数据库中已有 {len(existing_models_with_url)} 个模型带URL")
            log(f"⚡ 这些模型在列表页将跳过URL获取")
    except Exception as e:
        log(f"⚠️  无法加载已存在模型列表: {e}")
        log(f"🔄 将为所有模型获取URL")

    # 获取官方模型列表
    official_models_df = get_aistudio_official_models()
    if official_models_df is None or official_models_df.empty:
        log("❌ 没有找到AI Studio官方模型")
        return pd.DataFrame(), 0

    # 测试模式：只处理第一个模型
    if test_mode:
        official_models_df = official_models_df.head(1)
        log(f"🧪 测试模式：只处理第一个模型")

    # 创建AIStudioFetcher实例以复用_get_detailed_info方法
    fetcher = AIStudioFetcher(test_mode=test_mode, enable_detailed_log=False)

    driver = None
    all_derivative_models = []
    processed_count = 0
    total_count = len(official_models_df)
    skipped_url_count = 0  # 统计跳过URL获取的模型数

    try:
        driver = create_chrome_driver()

        for idx, row in official_models_df.iterrows():
            base_model_name = row['model_name']
            base_url = row['url']

            log(f"\n{'=' * 80}")
            log(f"[{idx + 1}/{total_count}] 处理模型: {base_model_name}")
            log(f"访问: {base_url}")
            driver.get(base_url)

            try:
                WebDriverWait(driver, SELENIUM_TIMEOUT).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(2)

                # 关闭广告横幅（每个模型页面关闭一次）
                try:
                    close_button_selectors = [
                        "#main > div.a-s-6th-footer-banner-wrapper > a > span",
                        "div.a-s-6th-footer-banner-wrapper > a > span",
                        ".a-s-6th-footer-banner-wrapper a span",
                    ]

                    for selector in close_button_selectors:
                        try:
                            close_buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                            if close_buttons:
                                close_buttons[0].click()
                                log(f"  ✅ 已关闭横幅广告")
                                time.sleep(0.5)
                                break
                        except:
                            continue

                    # 如果找不到关闭按钮，使用JavaScript移除
                    try:
                        driver.execute_script("""
                            var bannerWrapper = document.querySelector('div.a-s-6th-footer-banner-wrapper');
                            if (bannerWrapper) {
                                bannerWrapper.style.display = 'none';
                            }
                        """)
                    except:
                        pass

                except Exception as e:
                    # 关闭横幅失败不影响继续执行
                    pass

            except TimeoutException:
                log(f"⚠️  页面加载超时，跳过")
                continue

            # 查找模型血缘树元素
            try:
                tree_items = driver.find_elements(
                    By.CSS_SELECTOR,
                    "div.model-lineage-tree-item-wrap.child-model"
                )

                if not tree_items:
                    log(f"  ⚪️  没有找到衍生类型")
                    continue

                log(f"  ✅ 找到 {len(tree_items)} 个衍生类型")

                # 步骤2：先收集所有衍生类型的信息（避免stale element reference）
                tree_type_list = []
                for tree_item in tree_items:
                    try:
                        # 检查是否为"当前模型"标记（说明当前模型本身是衍生版本，不需要爬取）
                        try:
                            opt_current_elements = tree_item.find_elements(By.CSS_SELECTOR, "div.opt-current")
                            if opt_current_elements:
                                # 这是一个"当前模型"标记，跳过
                                try:
                                    skip_name_zh = tree_item.find_element(By.CSS_SELECTOR, "div.name-zh").text.strip()
                                    skip_name_en = tree_item.find_element(By.CSS_SELECTOR, "div.name-en").text.strip()
                                    log(f"  ⏭️  跳过 '{skip_name_zh} / {skip_name_en}'（当前模型本身是衍生版本）")
                                except:
                                    log(f"  ⏭️  跳过一个衍生类型（当前模型本身是衍生版本）")
                                continue
                        except:
                            pass

                        # 提取类型信息
                        name_zh = tree_item.find_element(
                            By.CSS_SELECTOR, "div.name-zh"
                        ).text.strip()

                        name_en = tree_item.find_element(
                            By.CSS_SELECTOR, "div.name-en"
                        ).text.strip()

                        # 提取模型数量
                        count_text = tree_item.find_element(
                            By.CSS_SELECTOR, "div.opt-link"
                        ).text.strip()

                        count_match = re.search(r'(\d+)', count_text)
                        count = int(count_match.group(1)) if count_match else 0

                        # 获取链接
                        link_element = tree_item.find_element(
                            By.CSS_SELECTOR, "a.model-lineage-tree-item"
                        )
                        link = link_element.get_attribute('href')

                        tree_type_list.append({
                            'name_zh': name_zh,
                            'name_en': name_en,
                            'count': count,
                            'link': link
                        })
                    except Exception as e:
                        log(f"  ⚠️  提取衍生类型信息时出错: {e}")
                        continue

                # 步骤3：对每个衍生类型获取模型列表
                for idx, tree_type in enumerate(tree_type_list):
                    try:
                        name_zh = tree_type['name_zh']
                        name_en = tree_type['name_en']
                        count = tree_type['count']
                        link = tree_type['link']

                        log(f"\n  📂 衍生类型: {name_zh} / {name_en} ({count}个模型)")

                        if link.startswith('/'):
                            full_url = f"https://aistudio.baidu.com{link}"
                        else:
                            full_url = link

                        # 访问衍生模型列表页
                        driver.get(full_url)

                        try:
                            WebDriverWait(driver, SELENIUM_TIMEOUT).until(
                                EC.presence_of_element_located(
                                    (By.CSS_SELECTOR, "div.ai-model-list-wapper")
                                )
                            )
                            time.sleep(2)
                        except TimeoutException:
                            log(f"    ⚠️  衍生模型列表页加载超时")
                            continue

                        # 提取所有模型卡片
                        cards = driver.find_elements(
                            By.CSS_SELECTOR,
                            "div.ai-model-list-wapper > div"
                        )

                        log(f"    ✅ 找到 {len(cards)} 个模型")

                        # 🔧 修复stale element reference：先提取所有卡片数据，再处理
                        card_data_list = []
                        for card_idx, card in enumerate(cards):
                            try:
                                # 获取模型名称
                                full_model_name = card.find_element(
                                    By.CSS_SELECTOR,
                                    "div.ai-model-list-wapper-card-right-desc"
                                ).text.strip()

                                # 获取发布者
                                publisher = card.find_element(
                                    By.CSS_SELECTOR,
                                    "span.ai-model-list-wapper-card-right-detail-one-publisher"
                                ).text.strip()

                                # 获取下载量和时间字段
                                detail_items = card.find_elements(
                                    By.CSS_SELECTOR,
                                    "div.ai-model-list-wapper-card-right-detail-one-item-tip"
                                )

                                usage_count = detail_items[0].find_element(
                                    By.CSS_SELECTOR,
                                    "span.ai-model-list-wapper-card-right-detail-one-like"
                                ).text.strip()

                                # 🔧 新增：获取更新时间（第3个tip）
                                last_modified = None
                                if len(detail_items) >= 3:
                                    try:
                                        last_modified = detail_items[2].find_element(
                                            By.CSS_SELECTOR,
                                            "span.ai-model-list-wapper-card-right-detail-one-like"
                                        ).text.strip()
                                    except:
                                        pass

                                # 保存卡片数据，避免stale element reference
                                card_data_list.append({
                                    'full_model_name': full_model_name,
                                    'publisher': publisher,
                                    'usage_count': usage_count,
                                    'last_modified': last_modified,
                                    'card_element': card  # 保存元素引用用于后续获取URL
                                })

                            except Exception as e:
                                log(f"      ⚠️  提取卡片信息时出错: {e}")
                                continue

                        # 处理所有卡片数据
                        for idx, card_data in enumerate(card_data_list):
                            try:
                                full_model_name = card_data['full_model_name']
                                publisher = card_data['publisher']
                                usage_count = card_data['usage_count']
                                last_modified = card_data['last_modified']
                                card = card_data['card_element']

                                # 处理模型名称
                                if full_model_name.startswith("PaddlePaddle/"):
                                    model_name = full_model_name[len("PaddlePaddle/"):]
                                else:
                                    model_name = full_model_name

                                # 检查模型是否已有URL（在search阶段已获取过）
                                model_key = f"{publisher}/{model_name}"
                                has_url = model_key in existing_models_with_url
                                # 🔧 修复：即使已有URL，如果列表页是简化格式，仍需获取精确值
                                needs_precise_count = fetcher._is_simplified_count(usage_count)
                                should_fetch_url = not has_url or needs_precise_count

                                if not should_fetch_url:
                                    log(f"      ⏭️  跳过URL获取（已有URL且列表页为精确值）: {model_key}")
                                    skipped_url_count += 1
                                    model_url = None
                                else:
                                    # 复用AIStudioFetcher的_get_detailed_info方法获取URL和/或精确下载量
                                    if has_url and needs_precise_count:
                                        log(f"      🔍 获取精确下载量（已有URL）: {usage_count}")
                                    elif not has_url:
                                        log(f"      🔍 获取URL和精确下载量: {model_key}")
                                    detailed_count, model_url = fetcher._get_detailed_info(
                                        driver, card, idx, list_usage_count=usage_count
                                    )
                                    if detailed_count:
                                        usage_count = detailed_count

                                # 创建记录
                                record = {
                                    'date': date.today().isoformat(),
                                    'repo': 'AI Studio',
                                    'model_name': model_name,
                                    'publisher': publisher,
                                    'download_count': usage_count,
                                    'model_category': classify_model(
                                        model_name,
                                        publisher,
                                        base_model_name
                                    ),
                                    'model_type': name_en.lower(),  # adapter, finetune, etc.
                                    'base_model': base_model_name,
                                    'data_source': 'model_tree',
                                    'search_keyword': base_model_name,
                                    'url': model_url,  # 从search或model tree获取的URL
                                    'last_modified': last_modified  # 🔧 新增：更新时间
                                }

                                all_derivative_models.append(record)

                            except Exception as e:
                                log(f"      ⚠️  处理模型时出错: {e}")
                                continue

                        # 返回基础模型详情页
                        driver.back()
                        time.sleep(1)

                    except Exception as e:
                        log(f"  ⚠️  处理衍生类型时出错: {e}")
                        continue

                processed_count += 1
                if progress_callback:
                    progress_callback(processed_count)

            except NoSuchElementException:
                log(f"  ⚪️  未找到模型血缘树元素")
                continue

        # 转换为DataFrame
        if all_derivative_models:
            df = pd.DataFrame(all_derivative_models)
            log(f"\n{'=' * 80}")
            log(f"✅ 成功获取 {len(df)} 个衍生模型")
            if skipped_url_count > 0:
                log(f"⚡ 跳过了 {skipped_url_count} 个已有URL的模型")
            log(f"{'=' * 80}")

            # 保存到数据库（如果需要）
            if save_to_db and not df.empty:
                try:
                    from ..db import save_to_db as save_to_db_func
                    save_to_db_func(df, DB_PATH)
                    log(f"💾 已保存 {len(df)} 条记录到数据库")
                except Exception as e:
                    log(f"⚠️ 保存到数据库失败: {e}")

            return df, len(df)
        else:
            log(f"\n{'=' * 80}")
            log(f"⚠️  没有找到任何衍生模型")
            if skipped_url_count > 0:
                log(f"⚡ 跳过了 {skipped_url_count} 个已有URL的模型")
            log(f"{'=' * 80}")
            return pd.DataFrame(), 0

    except Exception as e:
        log(f"\n❌ 获取AI Studio Model Tree失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), 0

    finally:
        if driver:
            driver.quit()


def update_aistudio_model_tree(save_to_db=True, test_mode=False):
    """
    更新AI Studio Model Tree数据（包含去重处理）

    Args:
        save_to_db: 是否保存到数据库
        test_mode: 测试模式，只处理第一个模型

    Returns:
        tuple: (DataFrame, total_count) 衍生模型数据和数量
    """
    print("\n🔄 开始更新AI Studio Model Tree数据...")

    # 获取衍生模型
    df, total_count = fetch_aistudio_model_tree(test_mode=test_mode)

    if df.empty:
        print("⚠️ 没有获取到任何衍生模型数据")
        return df, 0

    # 去重处理：检查数据库中是否已存在相同的模型（根据publisher和model_name）
    if save_to_db:
        try:
            from ..db import load_data_from_db, save_to_db as save_to_db_func
            import sqlite3

            # 获取现有AI Studio数据
            conn = sqlite3.connect(DB_PATH)
            existing_query = """
                SELECT DISTINCT publisher, model_name
                FROM model_downloads
                WHERE repo = 'AI Studio'
            """
            existing_df = pd.read_sql_query(existing_query, conn)
            conn.close()

            if not existing_df.empty:
                # 创建已存在模型的集合
                existing_models = set(
                    f"{row['publisher']}/{row['model_name']}"
                    for _, row in existing_df.iterrows()
                )

                # 过滤掉已存在的模型
                df['model_key'] = df['publisher'] + '/' + df['model_name']
                new_df = df[~df['model_key'].isin(existing_models)].copy()
                new_df = new_df.drop(columns=['model_key'])

                print(f"📊 去重前: {len(df)} 条，去重后: {len(new_df)} 条")
                print(f"🗑️  过滤掉 {len(df) - len(new_df)} 条已存在的记录")

                if new_df.empty:
                    print("⚠️ 没有新的模型需要保存")
                    return df, 0

                df = new_df

            # 保存到数据库
            save_to_db_func(df, DB_PATH)
            print(f"💾 已保存 {len(df)} 条新记录到数据库")

        except Exception as e:
            print(f"❌ 保存数据时出错: {e}")
            import traceback
            traceback.print_exc()

    return df, total_count
