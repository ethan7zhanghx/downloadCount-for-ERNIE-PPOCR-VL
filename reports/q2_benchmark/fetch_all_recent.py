"""
抓取近两个月内发布的所有模型（无下载量过滤）
输出到 output/all_recent_YYYYMMDD.csv
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ernie_tracker.config import HF_TOKEN
from huggingface_hub import HfApi
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CUTOFF = datetime.now(timezone.utc) - timedelta(days=60)
TODAY = datetime.now().strftime("%Y%m%d")

api = HfApi(token=HF_TOKEN)

print(f"抓取截止日期: {CUTOFF.date()} 以后发布的模型")
print("按 created_at 倒序遍历，遇到超出时间窗口停止...")

records = []
count = 0

for model in api.list_models(sort="created_at", direction=-1, cardData=False):
    created = model.created_at
    if created is None:
        continue
    if created < CUTOFF:
        print(f"已到时间边界，停止。最后一条: {created.date()}")
        break

    count += 1
    records.append({
        "model_id": model.id,
        "publisher": model.id.split("/")[0] if "/" in model.id else "",
        "pipeline_tag": model.pipeline_tag or "",
        "downloads": model.downloads or 0,
        "likes": model.likes or 0,
        "created_at": created.isoformat(),
        "tags": ",".join(model.tags or []),
    })

    if count % 5000 == 0:
        print(f"  已处理 {count} 条...")

print(f"\n共抓取 {len(records)} 个模型")

df = pd.DataFrame(records)
out_path = OUTPUT_DIR / f"all_recent_{TODAY}.csv"
df.to_csv(out_path, index=False)
print(f"已保存到 {out_path}")

# 基本统计
print(f"\n下载量分布:")
print(df["downloads"].describe(percentiles=[.5, .75, .9, .95, .99]))
print(f"\n零下载量: {(df['downloads'] == 0).sum()} ({(df['downloads'] == 0).mean()*100:.1f}%)")
print(f"\npipeline_tag top 10:")
print(df["pipeline_tag"].value_counts().head(10))
