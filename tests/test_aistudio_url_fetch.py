"""测试 AI Studio URL 收集功能"""
import sys
sys.path.insert(0, '/Users/zhanghaoxin/Desktop/Baidu/DownloadData')

from ernie_tracker.fetchers.selenium import AIStudioFetcher

def test_aistudio_url_collection():
    """测试AI Studio fetcher的URL收集功能"""
    print("=" * 60)
    print("测试 AI Studio fetcher - URL收集")
    print("=" * 60)

    fetcher = AIStudioFetcher()

    try:
        # 直接调用 fetch
        df, count = fetcher.fetch()

        print(f"\n成功获取 {count} 个模型")

        if not df.empty:
            print(f"\nDataFrame列名: {list(df.columns)}")

            # 检查 url 列
            if 'url' in df.columns:
                url_count = df['url'].notna().sum()
                print(f"\n有 URL 的模型数量: {url_count}/{len(df)}")

                if url_count > 0:
                    print("\n前 10 个模型的 URL:")
                    urls_df = df[df['url'].notna()].head(10)
                    for idx, row in urls_df.iterrows():
                        print(f"  - {row['model_name']}")
                        print(f"    URL: {row['url']}")
                        print(f"    发布者: {row['publisher']}")
                        print()
                else:
                    print("❌ 没有获取到任何URL")
            else:
                print("❌ DataFrame 中没有 url 列")

            # 统计发布者
            if 'publisher' in df.columns:
                print("\n发布者统计:")
                publisher_counts = df['publisher'].value_counts()
                for publisher, count in publisher_counts.items():
                    print(f"  {publisher}: {count}")

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_aistudio_url_collection()
