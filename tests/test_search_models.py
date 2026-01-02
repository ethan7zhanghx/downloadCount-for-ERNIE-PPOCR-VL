"""
测试搜索 ERNIE-4.5 和 PaddleOCR-VL 模型
"""
from huggingface_hub import list_models, model_info

print("=" * 80)
print("搜索 ERNIE-4.5 模型")
print("=" * 80)

ernie_models = list(list_models(search="ERNIE-4.5", full=True, limit=10))
print(f"\n找到 {len(ernie_models)} 个模型（显示前10个）\n")

for i, m in enumerate(ernie_models[:10], 1):
    print(f"{i}. {m.id}")
    print(f"   - publisher: {m.id.split('/')[0]}")
    print(f"   - downloads (from list_models): {getattr(m, 'downloads', 'N/A')}")

    # 尝试获取详细信息
    try:
        info = model_info(m.id, expand=["downloadsAllTime"])
        downloads_all_time = getattr(info, 'downloads_all_time', 'N/A')
        print(f"   - downloads_all_time (from model_info): {downloads_all_time}")
    except Exception as e:
        print(f"   - downloads_all_time (from model_info): ERROR - {e}")

    print()

print("\n" + "=" * 80)
print("搜索 PaddleOCR-VL 模型")
print("=" * 80)

paddleocr_models = list(list_models(search="PaddleOCR-VL", full=True, limit=10))
print(f"\n找到 {len(paddleocr_models)} 个模型（显示前10个）\n")

for i, m in enumerate(paddleocr_models[:10], 1):
    print(f"{i}. {m.id}")
    print(f"   - publisher: {m.id.split('/')[0]}")
    print(f"   - downloads (from list_models): {getattr(m, 'downloads', 'N/A')}")

    # 尝试获取详细信息
    try:
        info = model_info(m.id, expand=["downloadsAllTime"])
        downloads_all_time = getattr(info, 'downloads_all_time', 'N/A')
        print(f"   - downloads_all_time (from model_info): {downloads_all_time}")
    except Exception as e:
        print(f"   - downloads_all_time (from model_info): ERROR - {e}")

    print()
