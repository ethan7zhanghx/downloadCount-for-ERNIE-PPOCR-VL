#!/usr/bin/env python3
"""
AI Studio 爬虫测试脚本 - 带详细日志和测试模式

功能：
1. 测试模式：每个页面只获取第一个和最后一个模型
2. 详细日志：记录所有操作和耗时，用于分析问题

使用方法：
    python3 tests/test_aistudio_with_log.py
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ernie_tracker.fetchers.selenium import AIStudioFetcher


def main():
    print("=" * 80)
    print("AI Studio 爬虫测试")
    print("=" * 80)
    print()
    print("🧪 测试模式：每个页面只获取第一个和最后一个模型")
    print("📝 详细日志：所有操作记录到日志文件")
    print()

    # 创建 fetcher，启用测试模式和详细日志
    fetcher = AIStudioFetcher(
        test_mode=True,           # 测试模式：每个页面只处理第一个和最后一个模型
        enable_detailed_log=True  # 启用详细日志
    )

    print("开始爬取...")
    print()

    # 执行爬取
    df, count = fetcher.fetch()

    print()
    print("=" * 80)
    print(f"✅ 爬取完成！共获取 {count} 个模型")
    print("=" * 80)
    print()
    print("📊 结果预览：")
    print(df.to_string())
    print()
    print("📝 日志文件：请查看当前目录下的 aistudio_crawl_*.log 文件")


if __name__ == "__main__":
    main()
