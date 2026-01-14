#!/usr/bin/env python3
"""
AI Studio Model Tree 测试脚本

测试从官方模型页面获取衍生模型信息

流程：
1. 访问官方模型详情页
2. 查找模型血缘树元素（.model-lineage-tree-item-wrap）
3. 提取衍生类型（name-en）
4. 点击进入衍生模型列表页
5. 提取所有模型信息
6. 标记衍生类型后入库
"""

import sys
import os
import time
import re
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# 配置
SELENIUM_TIMEOUT = 20


def create_chrome_driver():
    """创建Chrome Driver（使用自动管理）"""
    from ernie_tracker.utils import create_chrome_driver
    return create_chrome_driver()


def extract_numbers(text):
    """从文本中提取数字"""
    numbers = re.findall(r'\d+', text.replace(',', ''))
    return int(numbers[0]) if numbers else None


def get_model_tree_types(driver, model_url):
    """
    从模型详情页获取衍生类型列表

    Args:
        driver: WebDriver instance
        model_url: 模型详情页URL

    Returns:
        list: 衍生类型列表，格式：[{"type": "Adapter", "count": 12, "link": "..."}, ...]
    """
    print(f"\n访问模型详情页: {model_url}")
    driver.get(model_url)

    # 等待页面加载
    try:
        WebDriverWait(driver, SELENIUM_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)
        print("✅ 页面加载成功")
    except TimeoutException:
        print("❌ 页面加载超时")
        return []

    # 查找模型血缘树元素
    try:
        tree_items = driver.find_elements(By.CSS_SELECTOR, "div.model-lineage-tree-item-wrap.child-model")
        print(f"\n找到 {len(tree_items)} 个衍生类型")

        derivative_types = []

        for idx, item in enumerate(tree_items):
            try:
                # 提取类型名称（中英文）
                name_zh = item.find_element(By.CSS_SELECTOR, "div.name-zh").text.strip()
                name_en = item.find_element(By.CSS_SELECTOR, "div.name-en").text.strip()

                # 提取模型数量
                count_text = item.find_element(By.CSS_SELECTOR, "div.opt-link").text.strip()
                count_match = re.search(r'(\d+)', count_text)
                count = int(count_match.group(1)) if count_match else 0

                # 提取链接
                link_element = item.find_element(By.CSS_SELECTOR, "a.model-lineage-tree-item")
                link = link_element.get_attribute('href')

                print(f"\n  [{idx + 1}] {name_zh} / {name_en}")
                print(f"      模型数量: {count}")
                print(f"      链接: {link}")

                derivative_types.append({
                    "type_zh": name_zh,
                    "type_en": name_en,
                    "count": count,
                    "link": link
                })

            except Exception as e:
                print(f"  ❌ 处理第 {idx + 1} 个衍生类型时出错: {e}")
                continue

        return derivative_types

    except NoSuchElementException:
        print("❌ 未找到模型血缘树元素")
        return []


def fetch_derivative_models(driver, derivative_type, base_model_name):
    """
    获取某个衍生类型的所有模型

    Args:
        driver: WebDriver instance
        derivative_type: 衍生类型字典
        base_model_name: 基础模型名称

    Returns:
        list: 模型列表
    """
    type_en = derivative_type['type_en']
    link = derivative_type['link']

    print(f"\n{'=' * 60}")
    print(f"获取 {type_en} 类型的衍生模型")
    print(f"{'=' * 60}")

    # 构建完整URL
    if link.startswith('/'):
        full_url = f"https://aistudio.baidu.com{link}"
    else:
        full_url = link

    print(f"访问: {full_url}")
    driver.get(full_url)

    # 等待页面加载
    try:
        WebDriverWait(driver, SELENIUM_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.ai-model-list-wapper"))
        )
        time.sleep(2)
        print("✅ 衍生模型列表页加载成功")
    except TimeoutException:
        print("❌ 衍生模型列表页加载超时")
        return []

    # 提取所有模型卡片
    cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
    print(f"找到 {len(cards)} 个模型")

    models = []

    for idx, card in enumerate(cards):
        try:
            # 获取模型名称
            full_model_name = card.find_element(
                By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
            ).text.strip()

            # 获取发布者
            publisher = card.find_element(
                By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-publisher"
            ).text.strip()

            # 获取下载量
            detail_items = card.find_elements(
                By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-detail-one-item-tip"
            )
            usage_count = detail_items[0].find_element(
                By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-like"
            ).text.strip()

            print(f"\n  [{idx + 1}] {full_model_name}")
            print(f"      发布者: {publisher}")
            print(f"      下载量: {usage_count}")

            # 处理模型名称
            if full_model_name.startswith("PaddlePaddle/"):
                model_name = full_model_name[len("PaddlePaddle/"):]
            else:
                model_name = full_model_name

            models.append({
                "model_name": model_name,
                "publisher": publisher,
                "download_count": usage_count,
                "derivative_type": type_en,
                "base_model": base_model_name
            })

        except Exception as e:
            print(f"  ❌ 处理第 {idx + 1} 个模型时出错: {e}")
            continue

    return models


def main():
    """主函数"""
    print("=" * 80)
    print("AI Studio Model Tree 测试")
    print("=" * 80)

    # 测试模型URL（PaddleOCR-VL）
    test_model_url = "https://aistudio.baidu.com/modelsdetail/37507/intro"

    driver = None
    try:
        driver = create_chrome_driver()

        # 步骤1：获取衍生类型列表
        print("\n步骤1: 获取衍生类型列表")
        print("-" * 80)
        derivative_types = get_model_tree_types(driver, test_model_url)

        if not derivative_types:
            print("\n❌ 未找到任何衍生类型")
            return

        print(f"\n✅ 成功获取 {len(derivative_types)} 个衍生类型")

        # 步骤2：测试获取第一个衍生类型的模型
        if derivative_types:
            first_type = derivative_types[0]
            base_model_name = "PaddleOCR-VL"

            models = fetch_derivative_models(driver, first_type, base_model_name)

            print(f"\n{'=' * 80}")
            print(f"✅ 成功获取 {len(models)} 个 {first_type['type_en']} 类型的模型")
            print(f"{'=' * 80}")

            # 显示结果预览
            if models:
                print("\n📊 结果预览（前5个）:")
                for model in models[:5]:
                    print(f"  - {model['model_name']}")
                    print(f"    发布者: {model['publisher']}")
                    print(f"    下载量: {model['download_count']}")
                    print(f"    衍生类型: {model['derivative_type']}")
                    print(f"    基础模型: {model['base_model']}")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if driver:
            driver.quit()


if __name__ == "__main__":
    main()
