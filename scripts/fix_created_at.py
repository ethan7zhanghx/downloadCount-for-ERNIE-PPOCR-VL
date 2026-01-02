"""
补充 Excel 文件中的 created_at 和 last_modified 字段
"""
from huggingface_hub import model_info
import pandas as pd
import sys
from datetime import datetime

def get_model_dates(model_id: str):
    """获取模型的 created_at 和 last_modified"""
    try:
        info = model_info(model_id)

        created_at = None
        last_modified = None

        if hasattr(info, 'created_at') and info.created_at:
            created_at = info.created_at.isoformat() if hasattr(info.created_at, 'isoformat') else str(info.created_at)

        if hasattr(info, 'last_modified') and info.last_modified:
            last_modified = info.last_modified.isoformat() if hasattr(info.last_modified, 'isoformat') else str(info.last_modified)

        return created_at, last_modified
    except Exception as e:
        print(f"  ❌ 获取 {model_id} 失败: {e}")
        return None, None


def fix_excel_dates(excel_file: str):
    """修复 Excel 文件中的日期字段"""
    print(f"📂 读取文件: {excel_file}")

    # 读取所有 sheet
    xls = pd.ExcelFile(excel_file)
    sheet_names = xls.sheet_names

    print(f"✅ 找到 {len(sheet_names)} 个 sheet: {sheet_names}")

    # 修复每个 sheet
    updated_sheets = {}

    for sheet_name in sheet_names:
        print(f"\n{'='*80}")
        print(f"🔧 处理 sheet: {sheet_name}")
        print(f"{'='*80}")

        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # 跳过统计汇总表
        if sheet_name == '统计汇总':
            print(f"  ⏭️  跳过统计汇总表")
            updated_sheets[sheet_name] = df
            continue

        print(f"  📊 总记录数: {len(df)}")

        # 检查需要更新的记录
        need_update = df['created_at'].isna() | (df['created_at'] == '')
        update_count = need_update.sum()

        print(f"  🔍 需要更新 created_at 的记录: {update_count}")

        if update_count == 0:
            print(f"  ✅ 所有记录已有 created_at")
            updated_sheets[sheet_name] = df
            continue

        # 更新每条记录
        success_count = 0
        fail_count = 0

        for idx, row in df[need_update].iterrows():
            model_id = row['model_id']

            if pd.isna(model_id) or model_id == '':
                continue

            print(f"  [{idx+1}/{len(df)}] {model_id}")

            created_at, last_modified = get_model_dates(model_id)

            if created_at:
                df.at[idx, 'created_at'] = created_at
                df.at[idx, 'last_modified'] = last_modified if last_modified else df.at[idx, 'last_modified']
                success_count += 1
                print(f"     ✅ created_at: {created_at}")
            else:
                fail_count += 1

        print(f"\n  📈 更新成功: {success_count}")
        print(f"  ❌ 更新失败: {fail_count}")

        updated_sheets[sheet_name] = df

    # 保存更新后的文件
    print(f"\n{'='*80}")
    print(f"💾 保存更新后的文件")
    print(f"{'='*80}")

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        for sheet_name, df in updated_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✅ {sheet_name}: {len(df)} 行")

    print(f"\n✅ 文件已更新: {excel_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        # 默认使用最新的 ERNIE model tree 文件
        import glob
        files = sorted(glob.glob("ernie_model_tree_*.xlsx"), reverse=True)
        if files:
            excel_file = files[0]
            print(f"🔍 使用最新文件: {excel_file}")
        else:
            print("❌ 找不到 ernie_model_tree_*.xlsx 文件")
            print("用法: python fix_created_at.py <excel_file>")
            sys.exit(1)

    fix_excel_dates(excel_file)
