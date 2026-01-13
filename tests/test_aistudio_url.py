"""测试 AI Studio fetcher 的 URL 提取功能"""
import sys
sys.path.insert(0, '/Users/zhanghaoxin/Desktop/Baidu/DownloadData')

from ernie_tracker.fetchers.selenium import AIStudioFetcher

def test_aistudio_url_extraction():
    """测试 AI Studio URL 提取是否正常工作"""
    print("=" * 60)
    print("测试 AI Studio fetcher URL 提取...")
    print("=" * 60)

    fetcher = AIStudioFetcher()

    try:
        # 只爬取第一页，测试 URL 提取
        df, count = fetcher.fetch()

        print(f"\n✅ 成功获取 {count} 个模型")

        if not df.empty:
            print(f"\nDataFrame 列名: {list(df.columns)}")

            # 检查是否有 url 列
            if 'url' in df.columns:
                url_count = df['url'].notna().sum()
                print(f"✅ url 列存在，共 {url_count} 个模型有 URL")

                # 显示前 5 个模型的 URL
                print("\n前 5 个模型的 URL:")
                for idx, row in df.head(5).iterrows():
                    print(f"  - {row['model_name']}: {row['url']}")

                # 检查 URL 格式
                valid_urls = df[df['url'].str.contains('aistudio.baidu.com', na=False)]
                print(f"\n✅ 其中 {len(valid_urls)} 个 URL 格式正确")

                # 检查 last_modified 字段
                if 'last_modified' in df.columns:
                    last_modified_count = df['last_modified'].notna().sum()
                    print(f"✅ last_modified 列存在，共 {last_modified_count} 个模型有时间")
                else:
                    print("⚠️  没有 last_modified 列")

            else:
                print("❌ 没有 url 列")

            return True
        else:
            print("❌ DataFrame 为空")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_aistudio_url_extraction()
