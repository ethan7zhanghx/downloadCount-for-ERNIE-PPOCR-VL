"""
测试 HuggingFace API 的下载量获取
"""
from huggingface_hub import model_info, list_models

# 测试单个模型 - ERNIE-4.5-0.3B-PT
test_model_id = "PaddlePaddle/ERNIE-4.5-0.3B-PT"

print(f"=" * 80)
print(f"测试模型: {test_model_id}")
print(f"=" * 80)

# 方法1: 使用 expand 参数
print("\n方法1: 使用 expand=['downloadsAllTime'] 参数")
try:
    info1 = model_info(test_model_id, expand=["downloadsAllTime"])
    print(f"✅ API 调用成功")
    print(f"  - downloads_all_time: {getattr(info1, 'downloads_all_time', 'NOT FOUND')}")
    print(f"  - downloads: {getattr(info1, 'downloads', 'NOT FOUND')}")
    print(f"  - 所有包含'download'的属性:")
    for attr in dir(info1):
        if 'download' in attr.lower():
            print(f"    - {attr}: {getattr(info1, attr, 'N/A')}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 方法2: 不使用 expand 参数
print("\n方法2: 不使用 expand 参数")
try:
    info2 = model_info(test_model_id)
    print(f"✅ API 调用成功")
    print(f"  - downloads_all_time: {getattr(info2, 'downloads_all_time', 'NOT FOUND')}")
    print(f"  - downloads: {getattr(info2, 'downloads', 'NOT FOUND')}")
    print(f"  - 所有包含'download'的属性:")
    for attr in dir(info2):
        if 'download' in attr.lower():
            print(f"    - {attr}: {getattr(info2, attr, 'N/A')}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 方法3: 从搜索结果中获取
print("\n方法3: 从 list_models 搜索结果中获取")
try:
    models = list(list_models(search="ERNIE-4.5-0.3B-PT", full=True, limit=1))
    if models:
        m = models[0]
        print(f"✅ 搜索成功，找到模型: {m.id}")
        print(f"  - ModelInfo 对象的所有属性:")
        for attr in ['downloads', 'downloads_all_time', 'likes', 'tags']:
            print(f"    - {attr}: {getattr(m, attr, 'NOT FOUND')}")

        # 然后再调用 model_info 获取详情
        print(f"\n  再次调用 model_info 获取详情:")
        info3 = model_info(m.id, expand=["downloadsAllTime"])
        print(f"    - downloads_all_time: {getattr(info3, 'downloads_all_time', 'NOT FOUND')}")
        print(f"    - downloads: {getattr(info3, 'downloads', 'NOT FOUND')}")
    else:
        print(f"❌ 未找到模型")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
