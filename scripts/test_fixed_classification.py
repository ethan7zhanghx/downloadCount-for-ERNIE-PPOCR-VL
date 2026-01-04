"""
测试修复后的分类逻辑
"""
import sys
sys.path.insert(0, '..')

from ernie_tracker.fetchers.fetchers_modeltree import classify_model

# 模拟从 tags 中提取 base_model
def extract_base_from_tags(tags):
    """从 tags 中提取 base_model"""
    if not tags:
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith('base_model:'):
            parts = tag.split(':', 2)
            if len(parts) >= 2:
                candidate = parts[-1]
                if '/' in candidate and not candidate.startswith('license:'):
                    return candidate
    return None

# 测试数据
test_cases = [
    {
        'model_name': 'RysOCR',
        'publisher': 'some-user',
        'tags': ['peft', 'safetensors', 'ocr', 'lora', 'transformers', 'polish',
                 'base_model:PaddlePaddle/PaddleOCR-VL',
                 'base_model:adapter:PaddlePaddle/PaddleOCR-VL']
    },
    {
        'model_name': 'polish-ocr-lora-broken',
        'publisher': 'some-user',
        'tags': ['peft', 'safetensors',
                 'base_model:adapter:PaddlePaddle/PaddleOCR-VL',
                 'lora', 'transformers', 'ocr',
                 'base_model:PaddlePaddle/PaddleOCR-VL']
    },
    {
        'model_name': 'PaddleOCR-VL-half-GGUF-pured',
        'publisher': 'some-user',
        'tags': ['quantized', 'gguf', 'base_model:PaddlePaddle/PaddleOCR-VL']
    },
]

print("=" * 80)
print("测试修复后的分类逻辑")
print("=" * 80)

for test in test_cases:
    # 从 tags 中提取 base_model
    base_from_tags = extract_base_from_tags(test['tags'])

    result = classify_model(
        model_name=test['model_name'],
        publisher=test['publisher'],
        base_model=base_from_tags
    )

    print(f"\n模型: {test['model_name']}")
    print(f"  从 tags 提取的 base_model: {base_from_tags}")
    print(f"  分类结果: {result}")

    expected = 'paddleocr-vl' if base_from_tags and 'paddleocr' in base_from_tags.lower() else 'ernie-4.5'

    if result != expected:
        print(f"  ❌ 错误! 应该是: {expected}")
    else:
        print(f"  ✅ 正确")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
