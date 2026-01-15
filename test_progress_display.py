#!/usr/bin/env python3
"""
测试进度显示功能
验证上周模型数量查询和进度计算
"""
import sys
from datetime import date, timedelta

from ernie_tracker.db import (
    init_database,
    get_previous_week_model_count,
    get_last_model_count,
)


def get_available_dates():
    """获取数据库中可用的日期列表"""
    import sqlite3
    from ernie_tracker.config import DB_PATH, DATA_TABLE

    init_database()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT date FROM {DATA_TABLE} ORDER BY date DESC")
    dates = [row[0] for row in cur.fetchall()]
    conn.close()
    return dates


def test_previous_week_count():
    """测试获取上周模型数量功能"""
    print("=" * 60)
    print("测试：获取上周模型数量")
    print("=" * 60)

    # 初始化数据库
    init_database()

    # 测试平台列表
    platforms = [
        "Hugging Face",
        "ModelScope",
        "AI Studio",
        "GitCode",
        "Modelers",
        "魔乐",
        "Gitee"
    ]

    print(f"\n当前日期: {date.today().isoformat()}")
    print(f"上周日期: {(date.today() - timedelta(days=7)).isoformat()}\n")

    for platform in platforms:
        # 获取上周模型数量
        week_count = get_previous_week_model_count(platform, days_ago=7)

        # 获取上次记录数量
        last_count = get_last_model_count(platform)

        # 确定使用的参考值
        reference_count = week_count if week_count else last_count

        print(f"平台: {platform}")
        print(f"  上周模型数量: {week_count if week_count else '无数据'}")
        print(f"  上次记录数量: {last_count if last_count else '无数据'}")
        print(f"  进度参考值: {reference_count if reference_count else '无数据（首次运行）'}")
        print()

    print("=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


def test_available_dates():
    """测试可用日期查询"""
    print("\n获取数据库中的可用日期:")
    print("-" * 40)

    dates = get_available_dates()
    if dates:
        for i, d in enumerate(dates[-10:], 1):  # 显示最近10个日期
            print(f"{i}. {d}")
    else:
        print("暂无数据")


if __name__ == "__main__":
    try:
        test_previous_week_count()
        test_available_dates()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
