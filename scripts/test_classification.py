"""
测试模型分类逻辑
"""
import sys
sys.path.insert(0, '..')

from ernie_tracker.fetchers.fetchers_modeltree import classify_model

# 测试有问题的模型
test_cases = [
    {
        'model_name': 'RysOCR',
        'publisher': 'some-user',
        'base_model': 'PaddlePaddle/PaddleOCR-VL'
    },
    {
        'model_name': 'polish-ocr-lora-broken',
        'publisher': 'some-user',
        'base_model': 'PaddlePaddle/PaddleOCR-VL'
    },
    {
        'model_name': 'PaddleOCR-VL-half-GGUF-pured',
        'publisher': 'some-user',
        'base_model': 'PaddlePaddle/PaddleOCR-VL'
    },
    {
        'model_name': 'PaddleOCR-VL-MLX',
        'publisher': 'some-user',
        'base_model': None
    }
]

print("=" * 80)
print("测试模型分类逻辑")
print("=" * 80)

for test in test_cases:
    result = classify_model(
        model_name=test['model_name'],
        publisher=test['publisher'],
        base_model=test['base_model']
    )

    print(f"\n模型: {test['model_name']}")
    print(f"  base_model: {test['base_model']}")
    print(f"  分类结果: {result}")

    # 检查是否应该是 paddleocr-vl
    should_be_paddleocr = False
    if test['base_model']:
        base_lower = test['base_model'].lower()
        should_be_paddleocr = 'paddleocr' in base_lower and 'vl' in base_lower

    name_lower = test['model_name'].lower()
    should_be_paddleocr = should_be_paddleocr or ('paddleocr' in name_lower and 'vl' in name_lower)

    expected = 'paddleocr-vl' if should_be_paddleocr else 'ernie-4.5'

    if result != expected:
        print(f"  ❌ 错误! 应该是: {expected}")
    else:
        print(f"  ✅ 正确")
