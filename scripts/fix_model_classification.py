"""
修复数据库中错误的模型分类
根据 base_model 重新分类模型
"""
import sqlite3
import sys
sys.path.insert(0, '..')

from ernie_tracker.fetchers.fetchers_modeltree import classify_model

DB_PATH = "../ernie_downloads.db"

def fix_model_classification():
    """修复数据库中的模型分类"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 80)
    print("开始修复模型分类")
    print("=" * 80)

    # 1. 找出所有需要修复的记录
    print("\n📊 分析需要修复的记录...")

    # 情况1：base_model 包含 PaddleOCR-VL，但被分类为 ernie-4.5
    cursor.execute("""
        SELECT COUNT(*)
        FROM model_downloads
        WHERE base_model LIKE '%PaddleOCR-VL%'
        AND model_category = 'ernie-4.5'
    """)
    count1 = cursor.fetchone()[0]
    print(f"  情况1：base_model 是 PaddleOCR-VL，但分类为 ernie-4.5：{count1} 条")

    # 情况2：base_model 包含 ERNIE，模型名不包含 PaddleOCR，但被分类为 paddleocr-vl
    # 排除官方 PaddleOCR-VL 模型（它们虽然 base 是 ERNIE，但应该归类为 paddleocr-vl）
    cursor.execute("""
        SELECT COUNT(*)
        FROM model_downloads
        WHERE (base_model LIKE '%ERNIE%' OR base_model LIKE '%ernie%')
        AND base_model NOT LIKE '%PaddleOCR%'
        AND model_category = 'paddleocr-vl'
        AND model_name NOT LIKE '%PaddleOCR%'
        AND model_name NOT LIKE '%paddleocr%'
    """)
    count2 = cursor.fetchone()[0]
    print(f"  情况2：base_model 是 ERNIE，但分类为 paddleocr-vl：{count2} 条")

    if count1 == 0 and count2 == 0:
        print("\n✅ 没有需要修复的记录！")
        conn.close()
        return

    # 2. 修复情况1：base_model 是 PaddleOCR-VL → 应该分类为 paddleocr-vl
    print("\n🔧 修复情况1：base_model 包含 PaddleOCR-VL 的记录...")
    cursor.execute("""
        UPDATE model_downloads
        SET model_category = 'paddleocr-vl'
        WHERE base_model LIKE '%PaddleOCR-VL%'
        AND model_category = 'ernie-4.5'
    """)
    fixed1 = cursor.rowcount
    print(f"  ✅ 已修复 {fixed1} 条记录")

    # 3. 修复情况2：base_model 是 ERNIE，且模型名不包含 PaddleOCR → 应该分类为 ernie-4.5
    print("\n🔧 修复情况2：base_model 包含 ERNIE 的记录...")
    cursor.execute("""
        UPDATE model_downloads
        SET model_category = 'ernie-4.5'
        WHERE (base_model LIKE '%ERNIE%' OR base_model LIKE '%ernie%')
        AND base_model NOT LIKE '%PaddleOCR%'
        AND model_category = 'paddleocr-vl'
        AND model_name NOT LIKE '%PaddleOCR%'
        AND model_name NOT LIKE '%paddleocr%'
    """)
    fixed2 = cursor.rowcount
    print(f"  ✅ 已修复 {fixed2} 条记录")

    # 提交更改
    conn.commit()

    print("\n" + "=" * 80)
    print(f"修复完成！共修复 {fixed1 + fixed2} 条记录")
    print("=" * 80)

    # 4. 验证修复结果
    print("\n📊 验证修复结果...")

    cursor.execute("""
        SELECT COUNT(*)
        FROM model_downloads
        WHERE base_model LIKE '%PaddleOCR-VL%'
        AND model_category = 'ernie-4.5'
    """)
    remaining1 = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM model_downloads
        WHERE (base_model LIKE '%ERNIE%' OR base_model LIKE '%ernie%')
        AND base_model NOT LIKE '%PaddleOCR%'
        AND model_category = 'paddleocr-vl'
        AND model_name NOT LIKE '%PaddleOCR%'
        AND model_name NOT LIKE '%paddleocr%'
    """)
    remaining2 = cursor.fetchone()[0]

    if remaining1 == 0 and remaining2 == 0:
        print("  ✅ 所有错误分类已修复！")
    else:
        print(f"  ⚠️ 仍有 {remaining1 + remaining2} 条记录未修复")

    conn.close()

if __name__ == "__main__":
    fix_model_classification()
