#!/usr/bin/env python3
"""
快速验证脚本：检查所有fetcher代码中是否包含URL字段
"""
import re

def check_file_for_url(filepath, patterns):
    """检查文件中是否包含URL字段的特定模式"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            results = {}
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, content, re.MULTILINE)
                results[pattern_name] = len(matches) > 0
            return results
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 80)
    print("URL字段代码验证")
    print("=" * 80)

    checks = {
        "HuggingFace (传统搜索)": {
            "file": "ernie_tracker/fetchers/fetchers_unified.py",
            "patterns": {
                "URL字段": r'"url":\s*f"https://huggingface\.co/\{m\.id\}"',
            }
        },
        "HuggingFace (Model Tree)": {
            "file": "ernie_tracker/fetchers/fetchers_modeltree.py",
            "patterns": {
                "URL字段": r"'url':\s*f'https://huggingface\.co/\{model_id\}'",
            }
        },
        "ModelScope": {
            "file": "ernie_tracker/fetchers/fetchers_unified.py",
            "patterns": {
                "URL字段": r'"url":\s*f"https://modelscope\.cn/models/\{model_id\}"',
            }
        },
        "GitCode": {
            "file": "ernie_tracker/fetchers/fetchers_unified.py",
            "patterns": {
                "URL字段": r'"url":\s*model_link',
            }
        },
        "CAICT": {
            "file": "ernie_tracker/fetchers/fetchers_unified.py",
            "patterns": {
                "URL字段": r'"url":\s*model_link',
            }
        },
        "AI Studio": {
            "file": "ernie_tracker/fetchers/selenium.py",
            "patterns": {
                "URL参数": r'url=model_url',
            }
        },
        "Gitee": {
            "file": "ernie_tracker/fetchers/selenium.py",
            "patterns": {
                "URL获取": r'model_url\s*=\s*link\.get_attribute\([\'"]href[\'"]\)',
                "URL参数": r'url=model_url',
            }
        },
        "Modelers": {
            "file": "ernie_tracker/fetchers/selenium.py",
            "patterns": {
                "URL获取": r'model_url\s*=\s*card\.get_attribute\([\'"]href[\'"]\)',
                "URL参数": r'url=model_url',
            }
        },
    }

    all_passed = True
    for platform, check in checks.items():
        filepath = check["file"]
        patterns = check["patterns"]

        print(f"\n{platform}:")
        results = check_file_for_url(filepath, patterns)

        if "error" in results:
            print(f"  ❌ 错误: {results['error']}")
            all_passed = False
            continue

        platform_passed = True
        for pattern_name, found in results.items():
            status = "✅" if found else "❌"
            print(f"  {status} {pattern_name}")
            if not found:
                platform_passed = False
                all_passed = False

        if platform_passed:
            print(f"  ✅ {platform} 通过")
        else:
            print(f"  ❌ {platform} 未完全通过")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有平台URL字段验证通过！")
    else:
        print("⚠️  部分平台可能需要检查")
    print("=" * 80)

    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
