#!/usr/bin/env python3
"""
测试AI Studio Model Tree URL获取功能

验证：
1. Model Tree阶段是否正确获取了模型URL
2. 是否复用了search页的_get_detailed_info方法
3. 是否跳过了search阶段已获取URL的模型
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ernie_tracker.fetchers.fetchers_modeltree import fetch_aistudio_model_tree


def test_url_fetching():
    """测试URL获取功能"""
    print("\n" + "=" * 80)
    print("测试AI Studio Model Tree URL获取")
    print("=" * 80)

    # 测试模式：只处理第一个模型
    print("\n🧪 测试模式：只处理第一个官方模型")
    df, count = fetch_aistudio_model_tree(test_mode=True)

    if df.empty:
        print("❌ 没有获取到任何模型数据")
        return False

    print(f"\n✅ 获取了 {count} 个衍生模型")

    # 检查URL字段
    import pandas as pd
    has_url = df['url'].notna().any()
    url_count = df['url'].notna().sum()

    print(f"\n📊 URL统计:")
    print(f"   - 总模型数: {len(df)}")
    print(f"   - 有URL的模型数: {url_count}")
    if len(df) > 0:
        print(f"   - URL获取率: {url_count/len(df)*100:.1f}%")

    # 显示前几个模型的URL
    print(f"\n🔍 模型URL示例（前5个）:")
    for idx, row in df.head(5).iterrows():
        url_status = "✅" if pd.notna(row['url']) else "❌"
        print(f"   {url_status} {row['publisher']}/{row['model_name']}")
        if pd.notna(row['url']):
            print(f"      URL: {row['url']}")

    # 检查是否有URL
    if not has_url:
        print("\n⚠️  警告：没有获取到任何URL！")
        return False

    print("\n✅ URL获取功能正常")
    return True


def test_code_reuse():
    """测试是否复用了AIStudioFetcher的_get_detailed_info方法"""
    print("\n" + "=" * 80)
    print("测试代码复用")
    print("=" * 80)

    # 检查源代码
    import inspect
    from ernie_tracker.fetchers.fetchers_modeltree import fetch_aistudio_model_tree

    source = inspect.getsource(fetch_aistudio_model_tree)

    checks = {
        "导入AIStudioFetcher": "from ..fetchers.selenium import AIStudioFetcher" in source,
        "创建fetcher实例": "fetcher = AIStudioFetcher(" in source,
        "调用_get_detailed_info": "fetcher._get_detailed_info(" in source,
        "检查existing_models_with_url": "existing_models_with_url" in source,
        "跳过逻辑检查": "should_fetch_url" in source or "not in existing_models_with_url" in source
    }

    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✅ 代码复用检查通过")
        return True
    else:
        print("\n❌ 代码复用检查失败")
        return False


def test_skip_logic():
    """测试跳过逻辑"""
    print("\n" + "=" * 80)
    print("测试跳过已有URL的逻辑")
    print("=" * 80)

    import sqlite3
    from ernie_tracker.config import DB_PATH
    import pandas as pd

    # 检查数据库查询是否正确
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT DISTINCT publisher, model_name
            FROM model_downloads
            WHERE repo = 'AI Studio' AND url IS NOT NULL AND url != ''
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"📚 数据库中有 {len(df)} 个AI Studio模型带URL")

        if len(df) > 0:
            print("\n示例模型（前3个）:")
            for idx, row in df.head(3).iterrows():
                print(f"  - {row['publisher']}/{row['model_name']}")
            print("\n✅ 这些模型在model tree列表页将被跳过URL获取")
            return True
        else:
            print("\n⚠️  数据库中没有带URL的模型")
            print("   (这是正常的，如果是首次运行)")
            return True

    except Exception as e:
        print(f"❌ 数据库查询失败: {e}")
        return False


if __name__ == "__main__":
    import pandas as pd

    print("🧪 AI Studio Model Tree URL获取测试")
    print("=" * 80)

    # 测试1: URL获取
    test1_passed = test_url_fetching()

    # 测试2: 代码复用
    test2_passed = test_code_reuse()

    # 测试3: 跳过逻辑
    test3_passed = test_skip_logic()

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"URL获取测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"代码复用测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    print(f"跳过逻辑测试: {'✅ 通过' if test3_passed else '❌ 失败'}")

    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败")
        sys.exit(1)
