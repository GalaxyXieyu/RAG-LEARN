"""
检索 + LLM 打标 + 分文档导出 CSV（简化参数 + 统一适配器）
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import csv
from datetime import datetime
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymilvus import connections
from finturn_embeding.config import RETRIEVAL_RESULTS_DIR, OCR_CHUNKS_DIR
from finturn_embeding.utils.embedder import get_embedder
from finturn_embeding.utils.llm import judge_factor_relevance as llm_judge_factor_relevance
from finturn_embeding.utils.chunks_io import lookup_content
from finturn_embeding.vector_db.milvus_adapter import MilvusAdapter

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


def _search_with_local_model(
    collection_name: str,
    query: str,
    embedder,
    limit: int = 10,
    expr: Optional[str] = None,
    output_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if output_fields is None:
        output_fields = [
            "chunk_id",
            "document_id",
            "text",
            "stage",
            "page_idx",
            "source_file",
            "headers_norm",
        ]
    query_vec = embedder.encode([query])[0]
    adapter = MilvusAdapter()
    hits = adapter.search(
        collection_name=collection_name,
        query_vec=query_vec,
        limit=limit,
        expr=expr,
        output_fields=output_fields,
        anns_field="dense_vector",
        metric_type="IP",
    )
    for h in hits:
        h["query"] = query
    return hits


def test_retrieval(
    collection_name: str,
    model_dir: Path,
    queries: List[str],
    cuda_device: Optional[int] = None,
    limit: int = 10,
    expr: Optional[str] = None,
    output_dir: Optional[Path] = None,
    enable_labeling: bool = True,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_concurrency: int = 5,
    chunks_dir: Optional[Path] = None,
    use_content_field: bool = True,
    query_with_stage: bool = False,
    stage_filter: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 60)
    print("【检索】Qwen3-Embedding-0.6B + LLM 打标（简化参数）")
    print("=" * 60)

    try:
        connections.connect(host="127.0.0.1", port="19530", timeout=10)
        print(f"✅ Milvus 连接成功，集合: {collection_name}")
    except Exception as e:
        print(f"❌ Milvus 连接失败: {e}")
        return {"total": 0, "success": 0, "failed": 0}

    try:
        embedder = get_embedder(Path(model_dir), cuda_device=cuda_device)
        print(f"✅ Embedding 模型加载成功 (cuda_device={cuda_device})")
    except Exception as e:
        print(f"❌ Embedding 模型加载失败: {e}")
        return {"total": 0, "success": 0, "failed": 0}

    all_results: List[Dict[str, Any]] = []
    success = 0
    failed = 0

    search_expr = expr
    if stage_filter:
        search_expr = f'({expr}) and (stage == "{stage_filter}")' if expr else f'stage == "{stage_filter}"'

    for i, query in enumerate(queries, 1):
        enhanced = f"{stage_filter} {query}" if (query_with_stage and stage_filter) else query
        print(f"\n[{i}/{len(queries)}] 查询: {enhanced}")
        if stage_filter:
            print(f"  阶段筛选: {stage_filter}")
        try:
            results = _search_with_local_model(
                collection_name=collection_name,
                query=enhanced,
                embedder=embedder,
                limit=limit,
                expr=search_expr,
            )
            if results:
                for r in results:
                    r["original_query"] = query
                all_results.extend(results)
                success += 1
                print(f"  命中: {len(results)} 条")
            else:
                print("  未检索到结果")
                failed += 1
        except Exception as e:
            print(f"  ❌ 检索失败: {e}")
            failed += 1

    if use_content_field and chunks_dir and chunks_dir.exists():
        print("\n【content 反查】从 JSON 加载 markdown ...")
        cache = 0
        for r in all_results:
            cid = r.get("chunk_id", "")
            did = r.get("document_id", "")
            if cid and did:
                c = lookup_content(cid, did, chunks_dir)
                if c:
                    r["content"] = c
                    cache += 1
        print(f"  成功填充 content: {cache}")

    if enable_labeling and all_results and llm_base_url and llm_api_key and llm_model:
        print("\n【打标】LLM 判断相关性 ...")
        def label_one(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            q = item.get("original_query") or item.get("query") or ""
            content = item.get("content") or item.get("text") or ""
            headers_norm = item.get("headers_norm", "")
            if not (q and content):
                return None
            t = llm_judge_factor_relevance(
                query=q,
                content=content,
                headers_norm=headers_norm,
                base_url=llm_base_url,
                api_key=llm_api_key,
                model=llm_model,
            )
            item["type"] = t or "unknown"
            return item

        labeled: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(llm_concurrency or 1))) as ex:
            futures = {ex.submit(label_one, r): r for r in all_results}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="打标中"):
                try:
                    x = fut.result()
                    if x:
                        labeled.append(x)
                except Exception as e:
                    tqdm.write(f"  ⚠️ 打标失败: {e}")
        all_results = labeled
        pos = sum(1 for r in all_results if r.get("type") == "positive")
        neg = sum(1 for r in all_results if r.get("type") == "negative")
        print(f"  统计: positive={pos}, negative={neg}")
    elif enable_labeling:
        for r in all_results:
            r["type"] = ""

    if all_results and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        grouped = defaultdict(list)
        for r in all_results:
            grouped[r.get("document_id", "unknown")].append(r)
        fields = [
            "query",
            "original_query",
            "score",
            "chunk_id",
            "document_id",
            "text",
            "stage",
            "page_idx",
            "source_file",
            "headers_norm",
        ]
        if any(r.get("content") for r in all_results):
            fields.append("content")
        if any(r.get("type") for r in all_results):
            fields.append("type")
        saved = 0
        for doc_id, items in grouped.items():
            if not doc_id or doc_id == "unknown":
                continue
            out_csv = output_dir / f"doc_{doc_id}_labeled.csv"
            with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for it in items:
                    w.writerow({k: it.get(k, "") for k in fields})
            saved += 1
            print(f"  ✅ {doc_id}: {len(items)} 条 → {out_csv.name}")
        print(f"\n✅ 共生成 {saved} 个文档 CSV")

    print("\n【完成】")
    return {
        "total": len(queries),
        "success": success,
        "failed": failed,
        "total_results": len(all_results),
        "output_dir": str(output_dir) if output_dir else None,
    }


def main():
    parser = argparse.ArgumentParser(description="检索+打标（简化参数）")
    parser.add_argument("--queries", type=str, nargs="+", required=True, help="查询词列表")
    parser.add_argument("--output-dir", type=str, help="输出目录（默认在 RETRIEVAL_RESULTS_DIR）")
    parser.add_argument("--stage-filter", type=str, help="阶段筛选（可选）")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (RETRIEVAL_RESULTS_DIR / datetime.now().strftime("retrieval_labeled_%Y%m%d_%H%M%S"))
    res = test_retrieval(
        collection_name="projects_documents_chunks_v2",
        model_dir=Path("/data/xieyu/Teaching/RAG/09_Embedding_Finetune/Qwen/Qwen3-Embedding-0.6B"),
        queries=args.queries,
        cuda_device=3,
        limit=10,
        expr=None,
        output_dir=out_dir,
        enable_labeling=True,
        llm_base_url=os.getenv("LLM_BASE_URL", "https://llm.3qiao.vip:23436/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", "sk-T3bQTqP2jlTMzjXJqjf9j4rnSuxxLmzH6EFGMN3afEYG2pLi"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5-72b-instruct-awq"),
        llm_concurrency=5,
        chunks_dir=Path(OCR_CHUNKS_DIR),
        use_content_field=True,
        query_with_stage=False,
        stage_filter=args.stage_filter,
    )
    print(f"\n结果: {res}")


if __name__ == "__main__":
    main()
