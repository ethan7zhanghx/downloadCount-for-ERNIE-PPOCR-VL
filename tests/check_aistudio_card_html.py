"""检查AI Studio card的HTML结构，寻找model ID"""
import sys
sys.path.insert(0, '/Users/zhanghaoxin/Desktop/Baidu/DownloadData')

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ernie_tracker.utils import create_chrome_driver

driver = create_chrome_driver()
wait = WebDriverWait(driver, 20)

try:
    url = "https://aistudio.baidu.com/modelsoverview?sortBy=useCount&q=ERNIE-4.5"
    driver.get(url)

    wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "div.ai-model-list-wapper")
    ))

    cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
    print(f"找到 {len(cards)} 个卡片\n")

    # 检查第一个卡片的HTML
    card = cards[0]
    html = card.get_attribute('outerHTML')

    print("=" * 60)
    print("第一个卡片的HTML结构：")
    print("=" * 60)
    print(html)
    print("\n" + "=" * 60)

    # 检查是否有data-*属性
    print("\n检查所有属性：")
    attrs = driver.execute_script(
        "var items = {}; for (var i = 0; i < arguments[0].attributes.length; i++) { items[arguments[0].attributes[i].name] = arguments[0].attributes[i].value; } return items;",
        card
    )
    for attr, value in attrs.items():
        print(f"  {attr}: {value}")

    print("\n" + "=" * 60)

    # 检查是否有onclick属性
    onclick = card.get_attribute('onclick')
    if onclick:
        print(f"发现onclick属性: {onclick}")
    else:
        print("没有onclick属性")

    # 检查所有子元素
    print("\n所有子元素的标签和属性:")
    children = card.find_elements(By.XPATH, ".//*")
    for i, child in enumerate(children[:10]):  # 只看前10个子元素
        tag = child.tag_name
        try:
            id_attr = child.get_attribute('id')
            class_attr = child.get_attribute('class')
            href = child.get_attribute('href')
            onclick = child.get_attribute('onclick')
            data_attrs = driver.execute_script(
                "var items = {}; for (var i = 0; i < arguments[0].attributes.length; i++) { var attr = arguments[0].attributes[i].name; if (attr.startsWith('data-')) { items[attr] = arguments[0].attributes[i].value; } } return items;",
                child
            )
            print(f"  [{i}] <{tag}> id={id_attr}, class={class_attr}, href={href}, onclick={onclick}, data={data_attrs}")
        except:
            pass

    input("按回车退出...")

finally:
    driver.quit()
