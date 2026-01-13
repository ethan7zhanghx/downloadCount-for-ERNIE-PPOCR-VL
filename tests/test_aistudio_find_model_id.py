"""检查卡片HTML中是否包含模型ID"""
import sys
sys.path.insert(0, '/Users/zhanghaoxin/Desktop/Baidu/DownloadData')

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ernie_tracker.utils import create_chrome_driver

def test_find_model_id():
    """在卡片HTML中查找模型ID"""
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
            import time
            time.sleep(0.5)
            print("✅ 已关闭横幅")
        except:
            pass

        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
        card = cards[0]

        model_name = card.find_element(
            By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
        ).text.strip()
        print(f"测试卡片: {model_name}")

        # 获取完整HTML
        html = card.get_attribute('outerHTML')
        print(f"\n=== 卡片HTML长度: {len(html)} 字符 ===")

        # 方法1: 查找所有数字（可能是ID）
        import re
        numbers = re.findall(r'\d+', html)
        print(f"\n找到的数字: {numbers[:20]}")  # 只显示前20个

        # 方法2: 查找特定格式的ID (4-5位数字)
        possible_ids = re.findall(r'\b\d{4,5}\b', html)
        print(f"\n可能的模型ID (4-5位数字): {possible_ids}")

        # 方法3: 检查所有属性
        print("\n=== 所有属性 ===")
        all_attrs = driver.execute_script("""
            var items = {};
            for (var i = 0; i < arguments[0].attributes.length; i++) {
                items[arguments[0].attributes[i].name] = arguments[0].attributes[i].value;
            }
            return items;
        """, card)
        for attr, value in all_attrs.items():
            if 'id' in attr.lower() or 'data' in attr.lower():
                print(f"  {attr}: {value}")

        # 方法4: 检查所有子元素的属性
        print("\n=== 子元素中的ID相关属性 ===")
        children = card.find_elements(By.XPATH, ".//*")
        id_attrs = []
        for child in children[:30]:  # 只检查前30个子元素
            try:
                tag = child.tag_name
                attrs = driver.execute_script("""
                    var items = {};
                    for (var i = 0; i < arguments[0].attributes.length; i++) {
                        var attr = arguments[0].attributes[i];
                        if (attr.name.includes('id') || attr.name.includes('data')) {
                            items[attr.name] = attr.value;
                        }
                    }
                    return items;
                """, child)
                if attrs:
                    print(f"  <{tag}>: {attrs}")
                    id_attrs.append(attrs)
            except:
                pass

        # 方法5: 点击卡片获取真实URL，然后对比HTML
        print("\n=== 验证: 点击获取真实ID ===")
        card.click()
        import time
        time.sleep(2)
        detail_url = driver.current_url
        print(f"详情页URL: {detail_url}")

        # 提取真实ID
        real_id = re.search(r'/modelsdetail/(\d+)/', detail_url)
        if real_id:
            real_id = real_id.group(1)
            print(f"真实模型ID: {real_id}")

            # 检查这个ID是否在HTML中
            if real_id in html:
                print(f"✅ 找到了！模型ID {real_id} 在卡片HTML中！")

                # 找到它在哪里
                lines = html.split('\n')
                for i, line in enumerate(lines):
                    if real_id in line:
                        print(f"  在第{i+1}行找到: {line.strip()[:100]}...")
                        break
            else:
                print(f"❌ 模型ID {real_id} 不在卡片HTML中")

        input("\n按回车退出...")

    finally:
        driver.quit()

if __name__ == "__main__":
    test_find_model_id()
