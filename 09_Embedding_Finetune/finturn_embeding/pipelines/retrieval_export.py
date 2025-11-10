"""
简单的因子检索 + 导出流水线封装

目标：
- 供 step3 直接复用，减少在脚本中重复实现构建查询、检索与 CSV 导出的模板代码。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Any, Optional
from datetime import datetime
import csv

from finturn_embeding.config import RETRIEVAL_RESULTS_DIR


def build_queries_from_factors(factors: List[Dict[str, Any]]) -> List[str]:
    """从因子结构构建查询字符串列表。"""
    queries: List[str] = []
    for item in factors:
        factor = (item or {}).get("factor", {}) or {}
        name = str(factor.get("factor") or factor.get("mapping_key") or "").strip()
        if name:
            queries.append(name)
    return queries


def retrieve_simple_results(
    queries: List[str],
    search_fn: Callable[[str, int, Optional[str]], List[Dict[str, Any]]],
    limit: int = 10,
    project_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    对每个 query 调用 `search_fn` 检索，抽取文本字段，形成统一结果列表。

    search_fn 接口约定：search_fn(query, limit, project_code) -> List[Dict|Any]
    """
    results: List[Dict[str, Any]] = []
    for q in queries:
        if not q:
            continue
        chunks = search_fn(q, limit, project_code)
        for c in chunks or []:
            if isinstance(c, dict):
                txt = (c.get("text") or "").strip()
            else:
                txt = str(c).strip()
            if txt:
                results.append({"query": q, "chunk": txt})
    return results


def export_results_to_csv(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """将简单检索结果导出为 CSV，列为 [query, chunk]。"""
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RETRIEVAL_RESULTS_DIR / f"factor_retrieval_{ts}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "chunk"])
        writer.writeheader()
        for r in results:
            writer.writerow({"query": r.get("query", ""), "chunk": r.get("chunk", "")})
    return output_path
