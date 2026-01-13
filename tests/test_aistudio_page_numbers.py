"""测试直接点击页码按钮能否快速跳转"""
import sys
sys.path.insert(0, '/Users/zhanghaoxin/Desktop/Baidu/DownloadData')

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ernie_tracker.utils import create_chrome_driver
import time

def test_page_number_click():
    """测试能否直接点击页码按钮快速跳转"""
    driver = create_chrome_driver()
    wait = WebDriverWait(driver, 20)

    try:
        url = "https://aistudio.baidu.com/modelsoverview?sortBy=useCount&q=ERNIE-4.5"
        driver.get(url)

        wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.ai-model-list-wapper")
        ))

        # 关闭横幅
        try:
            close_button = driver.find_element(By.CSS_SELECTOR, "#main > div.a-s-6th-footer-banner-wrapper > a > span")
            close_button.click()
            time.sleep(0.5)
            print("✅ 已关闭横幅")
        except:
            print("⚠️  未找到横幅或关闭失败")

        # 获取第1页第一个模型
        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
        first_model_page1 = cards[0].find_element(
            By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
        ).text.strip()
        print(f"第1页第一个模型: {first_model_page1}")

        # 查找所有分页相关元素
        print("\n=== 分页元素分析 ===")

        # 获取所有li元素
        pagination_items = driver.find_elements(By.CSS_SELECTOR, "li.ant-pagination-item")
        print(f"找到 {len(pagination_items)} 个页码按钮")

        for i, item in enumerate(pagination_items):
            try:
                page_num = item.text.strip()
                is_active = "ant-pagination-item-active" in item.get_attribute("class")
                print(f"  页码{i+1}: {page_num} (当前页: {is_active})")
            except:
                pass

        # 测试1: 直接点击第2个页码按钮（应该是第2页）
        print("\n=== 测试: 直接点击页码2 ===")
        if len(pagination_items) >= 2:
            try:
                page2_button = pagination_items[1]
                page2_text = page2_button.text.strip()
                print(f"点击页码: {page2_text}")

                # 用JavaScript点击
                driver.execute_script("arguments[0].click();", page2_button)
                time.sleep(2)

                cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                first_model = cards[0].find_element(
                    By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                ).text.strip()
                print(f"点击后第一个模型: {first_model}")

                if first_model != first_model_page1:
                    print(f"✅ 直接点击页码按钮有效！")
                else:
                    print(f"❌ 点击页码按钮无效")

            except Exception as e:
                print(f"❌ 点击失败: {e}")

        # 测试2: 点击"下一页"到第3页，然后尝试直接点击"第1页"按钮
        print("\n=== 测试: 从第3页直接回到第1页 ===")
        try:
            # 先翻到第2页
            next_button = driver.find_element(By.CSS_SELECTOR, "li.ant-pagination-next button")
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(2)

            # 再翻到第3页
            next_button = driver.find_element(By.CSS_SELECTOR, "li.ant-pagination-next button")
            if next_button.is_enabled():
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(2)

            # 获取第3页第一个模型
            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
            first_model_page3 = cards[0].find_element(
                By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
            ).text.strip()
            print(f"第3页第一个模型: {first_model_page3}")

            # 现在直接点击第1页按钮
            pagination_items = driver.find_elements(By.CSS_SELECTOR, "li.ant-pagination-item")
            if len(pagination_items) >= 1:
                page1_button = pagination_items[0]
                print(f"点击第1页按钮...")
                driver.execute_script("arguments[0].click();", page1_button)
                time.sleep(2)

                cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                first_model = cards[0].find_element(
                    By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                ).text.strip()
                print(f"点击后第一个模型: {first_model}")

                if first_model == first_model_page1:
                    print(f"✅ 可以从任意页直接跳回第1页！")
                else:
                    print(f"❌ 跳转失败")

        except Exception as e:
            print(f"❌ 测试失败: {e}")

        # 测试3: 检查是否有"快速跳转"输入框
        print("\n=== 测试: 查找页码输入框 ===")
        try:
            page_jumper = driver.find_element(By.CSS_SELECTOR, "input.ant-pagination-options-quick-jumper")
            print(f"✅ 找到页码输入框！")
            print(f"  尝试输入页码 '5' 并回车...")

            page_jumper.clear()
            page_jumper.send_keys("5")
            time.sleep(0.5)
            from selenium.webdriver.common.keys import Keys
            page_jumper.send_keys(Keys.RETURN)
            time.sleep(2)

            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
            first_model = cards[0].find_element(
                By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
            ).text.strip()
            print(f"跳转后第一个模型: {first_model}")

            if first_model != first_model_page1:
                print(f"✅ 页码输入框有效！可以快速跳转！")

        except:
            print(f"❌ 未找到页码输入框")

        input("\n按回车退出...")

    finally:
        driver.quit()

if __name__ == "__main__":
    test_page_number_click()
