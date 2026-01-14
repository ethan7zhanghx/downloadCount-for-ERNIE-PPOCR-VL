#!/usr/bin/env python3
"""
AI Studio Model Tree 直接测试 - PaddleOCR-VL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
import pandas as pd
import time
import re

from ernie_tracker.utils import create_chrome_driver
from ernie_tracker.config import SELENIUM_TIMEOUT
from ernie_tracker.fetchers.fetchers_modeltree import classify_model

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_single_model():
    """测试单个模型的Model Tree"""

    print("=" * 80)
    print("测试 PaddleOCR-VL 的 Model Tree")
    print("=" * 80)

    base_model_name = "PaddleOCR-VL"
    base_url = "https://aistudio.baidu.com/modelsdetail/37507/intro"

    driver = None
    all_derivative_models = []

    try:
        driver = create_chrome_driver()

        print(f"\n访问: {base_url}")
        driver.get(base_url)

        WebDriverWait(driver, SELENIUM_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)

        # 查找模型血缘树元素
        tree_items = driver.find_elements(
            By.CSS_SELECTOR,
            "div.model-lineage-tree-item-wrap.child-model"
        )

        print(f"找到 {len(tree_items)} 个衍生类型")

        if not tree_items:
            print("❌ 没有找到衍生类型")
            return pd.DataFrame(), 0

        # 先收集所有衍生类型的信息（避免stale element reference）
        tree_type_list = []
        for idx, tree_item in enumerate(tree_items):
            try:
                name_zh = tree_item.find_element(By.CSS_SELECTOR, "div.name-zh").text.strip()
                name_en = tree_item.find_element(By.CSS_SELECTOR, "div.name-en").text.strip()

                count_text = tree_item.find_element(By.CSS_SELECTOR, "div.opt-link").text.strip()
                count_match = re.search(r'(\d+)', count_text)
                count = int(count_match.group(1)) if count_match else 0

                # 获取链接
                link_element = tree_item.find_element(By.CSS_SELECTOR, "a.model-lineage-tree-item")
                link = link_element.get_attribute('href')

                if link.startswith('/'):
                    full_url = f"https://aistudio.baidu.com{link}"
                else:
                    full_url = link

                tree_type_list.append({
                    'name_zh': name_zh,
                    'name_en': name_en,
                    'count': count,
                    'full_url': full_url
                })
            except Exception as e:
                print(f"  ⚠️  提取衍生类型信息时出错: {e}")
                continue

        # 处理每个衍生类型
        for idx, tree_type in enumerate(tree_type_list):
            try:
                name_zh = tree_type['name_zh']
                name_en = tree_type['name_en']
                count = tree_type['count']
                full_url = tree_type['full_url']

                print(f"\n[{idx + 1}] {name_zh} / {name_en} ({count}个模型)")
                print(f"  访问: {full_url}")
                driver.get(full_url)

                WebDriverWait(driver, SELENIUM_TIMEOUT).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.ai-model-list-wapper"))
                )
                time.sleep(2)

                # 提取模型卡片
                cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                print(f"  找到 {len(cards)} 个模型")

                for card_idx, card in enumerate(cards):
                    try:
                        full_model_name = card.find_element(
                            By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                        ).text.strip()

                        publisher = card.find_element(
                            By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-publisher"
                        ).text.strip()

                        detail_items = card.find_elements(
                            By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-detail-one-item-tip"
                        )

                        usage_count = detail_items[0].find_element(
                            By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-like"
                        ).text.strip()

                        if full_model_name.startswith("PaddlePaddle/"):
                            model_name = full_model_name[len("PaddlePaddle/"):]
                        else:
                            model_name = full_model_name

                        record = {
                            'date': date.today().isoformat(),
                            'repo': 'AI Studio',
                            'model_name': model_name,
                            'publisher': publisher,
                            'download_count': usage_count,
                            'model_category': classify_model(model_name, publisher, base_model_name),
                            'model_type': name_en.lower(),
                            'base_model': base_model_name,
                            'data_source': 'model_tree',
                            'search_keyword': base_model_name
                        }

                        all_derivative_models.append(record)

                        if card_idx < 3:  # 只打印前3个
                            print(f"    [{card_idx + 1}] {model_name} - {usage_count}")

                    except Exception as e:
                        print(f"      ⚠️  处理模型出错: {e}")
                        continue

                # 返回详情页
                driver.back()
                time.sleep(1)

            except Exception as e:
                print(f"  ⚠️  处理衍生类型时出错: {e}")
                continue

        # 转换为DataFrame
        if all_derivative_models:
            df = pd.DataFrame(all_derivative_models)
            print(f"\n{'=' * 80}")
            print(f"✅ 成功获取 {len(df)} 个衍生模型")
            print(f"{'=' * 80}")

            print("\n📊 衍生类型统计:")
            print(df['model_type'].value_counts())

            return df, len(df)
        else:
            print("\n❌ 没有获取到任何衍生模型")
            return pd.DataFrame(), 0

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), 0

    finally:
        if driver:
            input("\n按回车键关闭浏览器...")
            driver.quit()

if __name__ == "__main__":
    df, count = test_single_model()

    if not df.empty:
        print("\n📊 完整数据预览:")
        print(df[['model_name', 'publisher', 'download_count', 'model_type', 'base_model']].to_string())
