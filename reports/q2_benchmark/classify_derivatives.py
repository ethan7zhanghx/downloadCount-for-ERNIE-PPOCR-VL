"""
从近两个月全量模型里区分出衍生模型分析基础：
1. 有 base_model tag → 直接识别，提取 base_model
2. 无 base_model tag 但下载量 >= P90 → Claude 批量判断
3. 排除"自引用"：衍生模型 publisher == base_model publisher

输出：derivatives_base_YYYYMMDD.csv
"""

import sys
import re
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ernie_tracker.config import HF_TOKEN
import pandas as pd
INPUT_CSV = Path(__file__).parent / "output" / "all_recent_20260424.csv"
OUTPUT_DIR = Path(__file__).parent / "output"
TODAY = datetime.now().strftime("%Y%m%d")

P90_THRESHOLD = 149  # downloads

# ─── 辅助函数 ───────────────────────────────────────────────

def extract_base_model(tags_str):
    """从 tags 字段提取主 base_model（优先取 base_model:quantized: 或 base_model:finetune:，否则取 base_model:xxx）"""
    if not isinstance(tags_str, str):
        return None
    tags = tags_str.split(',')
    # 优先取带类型前缀的（更精确）
    typed = [t for t in tags if re.match(r'base_model:(quantized|finetune|adapter|lora|merge):', t)]
    plain = [t for t in tags if re.match(r'base_model:[^:]+/[^:]+$', t)]
    candidates = typed if typed else plain
    if not candidates:
        return None
    # 取第一个，去掉前缀，只保留 org/model 部分
    raw = candidates[0]
    raw = re.sub(r'^base_model:(quantized|finetune|adapter|lora|merge):', '', raw)
    raw = re.sub(r'^base_model:', '', raw)
    return raw.strip() if raw.strip() else None

def infer_model_type_from_tags(tags_str, model_id):
    """从 tags 推断量化/微调类型"""
    if not isinstance(tags_str, str):
        tags_str = ''
    mid = model_id.lower()
    tags_lower = tags_str.lower()

    if 'base_model:quantized:' in tags_lower:
        return 'quantized'
    if 'base_model:finetune:' in tags_lower or 'base_model:adapter:' in tags_lower:
        return 'finetune'
    if 'base_model:lora:' in tags_lower:
        return 'lora'
    if 'base_model:merge:' in tags_lower:
        return 'merge'
    # 名字模式兜底
    quant_pats = ['-gguf', '-gptq', '-awq', '-exl2', '-fp8', 'q4_', 'q5_', 'q8_', 'int4', 'int8', '-mlx-']
    if any(p in mid for p in quant_pats):
        return 'quantized'
    return 'finetune'  # 默认

def is_self_referential(model_id, base_model):
    """衍生 publisher 和 base_model publisher 相同则为自引用"""
    if not base_model or '/' not in model_id or '/' not in base_model:
        return False
    return model_id.split('/')[0].lower() == base_model.split('/')[0].lower()

# ─── Claude 批量判断 ───────────────────────────────────────────────

QUANT_NAME_PATTERNS = [
    '-gguf', '-gptq', '-awq', '-exl2', '-fp8', '-bnb', '-mlx',
    'q4_', 'q5_', 'q6_', 'q8_', 'q2_', 'q3_',
    'int4', 'int8', 'int2',
    '-4bit', '-8bit', '-2bit',
    'nf4', 'bf16-', '-quantized',
]
FINETUNE_NAME_PATTERNS = [
    '-lora', '-qlora', '-sft', '-finetuned', '-finetune',
    '-instruct-', '-chat-', '-dpo', '-rlhf', '-ppo',
    '-adapter', '-merged', '-abliterated',
]

def name_pattern_classify(model_id, pipeline_tag):
    """通过名字模式判断是否为衍生，不需要 Claude。"""
    mid = model_id.lower()
    # 量化特征词
    if any(p in mid for p in QUANT_NAME_PATTERNS):
        return True, 'quantized', 'name_pattern'
    # 微调特征词
    if any(p in mid for p in FINETUNE_NAME_PATTERNS):
        return True, 'finetune', 'name_pattern'
    # pipeline_tag 是 text-to-image 或 image-text-to-text 且有 lora/adapter 词
    if pipeline_tag in ('text-to-image',) and any(p in mid for p in ['-lora', 'lora-', '_lora']):
        return True, 'lora', 'name_pattern'
    return False, None, None

# ─── 主流程 ───────────────────────────────────────────────

print("加载数据...")
df = pd.read_csv(INPUT_CSV)
print(f"总量: {len(df):,}")

# Level 1: 有 base_model tag
df['base_model_extracted'] = df['tags'].apply(extract_base_model)
has_tag = df[df['base_model_extracted'].notna()].copy()
has_tag['how_identified'] = 'tag'
has_tag['claude_confidence'] = 'N/A'
print(f"Level 1 (有tag): {len(has_tag):,}")

# Level 2: 无tag，下载量 >= P90，用 Claude 判断
no_tag = df[df['base_model_extracted'].isna()].copy()
high_dl = no_tag[no_tag['downloads'] >= P90_THRESHOLD].copy()
print(f"Level 2 候选 (无tag, downloads>={P90_THRESHOLD}): {len(high_dl):,}")

print(f"开始名字模式判断 Level 2，共 {len(high_dl)} 条...")
matched_rows = []
for _, row in high_dl.iterrows():
    is_deriv, mtype, how = name_pattern_classify(row['model_id'], row['pipeline_tag'])
    if is_deriv:
        r = row.copy()
        r['base_model_extracted'] = None  # 无 tag，base model 未知
        r['how_identified'] = how
        r['claude_confidence'] = 'N/A'
        r['model_type_override'] = mtype
        matched_rows.append(r)

high_dl_derivatives = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame()
print(f"Level 2 名字模式判断为衍生: {len(high_dl_derivatives):,} / {len(high_dl):,}")

# 合并
combined = pd.concat([has_tag, high_dl_derivatives], ignore_index=True)

# 排除自引用
combined['is_self'] = combined.apply(
    lambda r: is_self_referential(r['model_id'], r['base_model_extracted']), axis=1)
before = len(combined)
combined = combined[~combined['is_self']].drop(columns=['is_self'])
print(f"排除自引用: {before - len(combined):,} 条")

# 推断 model_type（Level 2 有 override，Level 1 从 tags 推断）
def get_model_type(row):
    if 'model_type_override' in row and pd.notna(row.get('model_type_override')):
        return row['model_type_override']
    return infer_model_type_from_tags(row.get('tags', ''), row['model_id'])

combined['model_type'] = combined.apply(get_model_type, axis=1)

# 最终输出字段
out_cols = ['model_id', 'publisher', 'base_model_extracted', 'model_type',
            'pipeline_tag', 'downloads', 'likes', 'created_at',
            'how_identified', 'claude_confidence']
result = combined[out_cols].rename(columns={'base_model_extracted': 'base_model'})
result = result.sort_values('downloads', ascending=False).reset_index(drop=True)

out_path = OUTPUT_DIR / f"derivatives_base_{TODAY}.csv"
result.to_csv(out_path, index=False)
print(f"\n最终衍生模型基础表: {len(result):,} 条")
print(f"已保存到 {out_path}")

print(f"\n下载量分布:")
print(result['downloads'].describe(percentiles=[.5,.75,.9,.95,.99]))
print(f"\nmodel_type 分布:")
print(result['model_type'].value_counts())
print(f"\nhow_identified 分布:")
print(result['how_identified'].value_counts())
