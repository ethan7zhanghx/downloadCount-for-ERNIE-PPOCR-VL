"""Civitai 平台爬虫实现 - 使用 REST API"""
import socket
import requests
from .base_fetcher import BaseFetcher


# Civitai 上搜索 ERNIE 相关模型的关键词
CIVITAI_SEARCH_QUERIES = ["ERNIE"]

# Civitai API 基础 URL
CIVITAI_API_BASE = "https://civitai.com/api/v1"


def _is_ernie_base_model(base_model):
    """判断 version 的 baseModel 是否为 ERNIE 系列"""
    return base_model and base_model.lower() == 'ernie'


# Civitai type 直接用原始值转小写，不做映射
CIVITAI_TYPE_MAP = {}


class CivitaiFetcher(BaseFetcher):
    """Civitai 爬虫 - 通过 REST API 获取模型数据"""

    def __init__(self):
        super().__init__("Civitai")

    def _request_ipv4(self, url, params, timeout=30):
        """强制使用 IPv4 发起请求（避免 IPv6 连接被 reset）"""
        import urllib3.util.connection
        original = urllib3.util.connection.allowed_gai_family

        # 临时替换为仅 IPv4
        urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            urllib3.util.connection.allowed_gai_family = original

    def fetch(self, progress_callback=None, progress_total=None):
        """
        抓取 Civitai 上的 ERNIE 相关模型

        Civitai 一个 model 下可能有多个 version，且不同 version 的 baseModel 可能不同
        （如 RedCraft 同时有 Flux、SDXL、Ernie 版本）。因此按 version 级别统计，
        只计入 baseModel 为 "Ernie" 的版本的下载量。

        同一个 model 下如果有多个 ERNIE 版本，按 version 分别记录。
        """
        seen_ids = set()
        total_count = 0
        skipped_models = 0
        skipped_versions = 0

        for query in CIVITAI_SEARCH_QUERIES:
            print(f"[Civitai] 搜索关键词: {query}")
            cursor = None

            while True:
                params = {"query": query, "limit": 100}
                if cursor:
                    params["cursor"] = cursor

                try:
                    data = self._request_ipv4(f"{CIVITAI_API_BASE}/models", params)
                except requests.RequestException as e:
                    print(f"[Civitai] API 请求失败: {e}")
                    break

                items = data.get("items", [])
                if not items:
                    break

                for item in items:
                    model_id = item.get("id")
                    if model_id in seen_ids:
                        continue
                    seen_ids.add(model_id)

                    model_name = item.get("name", "")
                    creator = item.get("creator", {})
                    username = creator.get("username", "Unknown")
                    versions = item.get("modelVersions", [])

                    # 筛选 baseModel 为 ERNIE 的版本
                    ernie_versions = [
                        v for v in versions
                        if _is_ernie_base_model(v.get("baseModel", ""))
                    ]

                    if not ernie_versions:
                        skipped_models += 1
                        continue

                    skipped_versions += len(versions) - len(ernie_versions)
                    model_type_raw = item.get("type", "")
                    our_model_type = CIVITAI_TYPE_MAP.get(model_type_raw, model_type_raw.lower())

                    for v in ernie_versions:
                        version_name = v.get("name", "")
                        version_id = v.get("id", "")
                        v_stats = v.get("stats", {})
                        download_count = v_stats.get("downloadCount", 0)
                        published_at = v.get("publishedAt", "")

                        # 模型名：如果只有一个 ERNIE 版本，用 model 名
                        # 如果有多个，用 "model名 / version名" 区分
                        if len(ernie_versions) == 1:
                            display_name = model_name
                        else:
                            display_name = f"{model_name} / {version_name}"

                        url = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"

                        record = self.create_record(
                            model_name=display_name,
                            publisher=username,
                            download_count=download_count,
                            search_keyword=query,
                            url=url,
                            model_category='ernie-image',
                            created_at=published_at,
                        )
                        record["model_type"] = our_model_type
                        record["base_model"] = "ERNIE-Image"
                        self.results.append(record)

                        total_count += 1
                        print(f"[Civitai] {total_count}. [{our_model_type}] {display_name} (by {username}): {download_count}")

                if progress_callback:
                    progress_callback(total_count, discovered_total=total_count)

                # 翻页
                metadata = data.get("metadata", {})
                next_cursor = metadata.get("nextCursor")
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor

        print(f"[Civitai] 共找到 {total_count} 个 ERNIE 版本"
              f"（跳过 {skipped_models} 个无关模型，{skipped_versions} 个非 ERNIE 版本）")
        return self.to_dataframe(), total_count
