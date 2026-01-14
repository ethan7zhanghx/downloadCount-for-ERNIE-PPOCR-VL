#!/usr/bin/env python3
"""
AI Studio Model Tree 集成测试脚本
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ernie_tracker.fetchers.fetchers_modeltree import update_aistudio_model_tree

def main():
    print("=" * 80)
    print("AI Studio Model Tree 集成测试")
    print("=" * 80)
    print()
    print("🧪 测试模式：只处理第一个官方模型")
    print("💾 不保存到数据库")
    print()

    # 运行测试（test_mode=True, save_to_db=False）
    df, count = update_aistudio_model_tree(save_to_db=False, test_mode=True)

    print()
    print("=" * 80)
    print(f"✅ 测试完成！共获取 {count} 个衍生模型")
    print("=" * 80)

    if not df.empty:
        print()
        print("📊 结果预览（所有衍生模型）:")
        print(df[['model_name', 'publisher', 'download_count', 'model_type', 'base_model']].to_string())
        print()
        print("📊 统计信息:")
        print(f"  总数: {len(df)}")
        print(f"  衍生类型分布:")
        print(df['model_type'].value_counts())
        print()
        print("📊 基础模型分布:")
        print(df['base_model'].value_counts())

if __name__ == "__main__":
    main()
