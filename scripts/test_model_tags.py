"""
Quick helper to inspect Hugging Face tag fields for a single model.

Usage:
    python scripts/test_model_tags.py <model_id>

It prints tags from multiple sources (model_info base/expand, list_models)
so we can verify whether structured tags like base_model:adapter:* are present.
"""
import sys
from typing import Any, Iterable, Optional

from huggingface_hub import list_models, model_info


def _pick(sources: Iterable[Any], name: str) -> Optional[Any]:
    for src in sources:
        if src is None:
            continue
        if hasattr(src, name):
            val = getattr(src, name)
            if val is not None:
                return val
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_model_tags.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]
    print(f"🔍 Inspecting model: {model_id}\n")

    base_info = expand_info = files_info = None
    try:
        base_info = model_info(model_id)
    except Exception as e:
        print(f"⚠️ model_info base failed: {e}")
    try:
        expand_info = model_info(model_id, expand=["downloadsAllTime", "trendingScore"])
    except Exception as e:
        print(f"⚠️ model_info expand failed: {e}")

    model_obj = None
    try:
        models = list(list_models(model_name=model_id, full=True, limit=1))
        model_obj = models[0] if models else None
    except Exception as e:
        print(f"⚠️ list_models failed: {e}")

    info_sources = [expand_info, base_info]

    # Print tags from each source
    def fmt_tags(val):
        if val is None:
            return "None"
        return str(val)

    print("Tags:")
    print(f"  - base_info.tags:   {fmt_tags(_pick([base_info], 'tags'))}")
    print(f"  - expand_info.tags: {fmt_tags(_pick([expand_info], 'tags'))}")
    print(f"  - list_models.tags: {fmt_tags(_pick([model_obj], 'tags'))}")

    # Pipeline tag and base_model for context
    print("\nPipeline tag:")
    print(f"  - base_info:   {_pick([base_info], 'pipeline_tag')}")
    print(f"  - expand_info: {_pick([expand_info], 'pipeline_tag')}")
    print(f"  - list_models: {_pick([model_obj], 'pipeline_tag')}")

    # Basic downloads snapshot to confirm expand call worked
    print("\nDownloads:")
    print(f"  - downloads_all_time (expand): {_pick([expand_info], 'downloads_all_time')}")
    print(f"  - downloads (base):            {_pick([base_info], 'downloads')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
