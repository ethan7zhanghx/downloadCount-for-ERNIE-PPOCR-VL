"""简单测试 AI Studio fetcher - 只测试第一页"""
import sys
sys.path.insert(0, '/Users/zhanghaoxin/Desktop/Baidu/DownloadData')

from ernie_tracker.fetchers.selenium import AIStudioFetcher

def test_aistudio_first_page_only():
    """只测试第一页，避免多次重试"""
    print("=" * 60)
    print("测试 AI Studio fetcher - 第一页")
    print("=" * 60)

    fetcher = AIStudioFetcher()

    try:
        # 直接调用 fetch，让它完整运行
        df, count = fetcher.fetch()

        print(f"\n成功获取 {count} 个模型")

        if not df.empty:
            print(f"DataFrame列名: {list(df.columns)}")

            # 检查 url 列
            if 'url' in df.columns:
                url_count = df['url'].notna().sum()
                print(f"有 URL 的模型数量: {url_count}/{len(df)}")

                if url_count > 0:
                    print("\n前 5 个有 URL 的模型:")
                    urls_df = df[df['url'].notna()].head(5)
                    for idx, row in urls_df.iterrows():
                        print(f"  - {row['model_name']}: {row['url']}")
                else:
                    print("没有获取到任何URL")
            else:
                print("❌ DataFrame 中没有 url 列")

            # 检查 last_modified 列
            if 'last_modified' in df.columns:
                time_count = df['last_modified'].notna().sum()
                print(f"有 last_modified 的模型数量: {time_count}/{len(df)}")
            else:
                print("❌ DataFrame 中没有 last_modified 列")

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_aistudio_first_page_only()
