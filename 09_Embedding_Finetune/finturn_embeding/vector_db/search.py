"""
统一检索接口：dense/sparse/hybrid（面向 utils/embedding_helper 等调用）
当前实现仅保留 hybrid_search 封装，后续可扩展。
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from .milvus_adapter import MilvusAdapter


def hybrid_search(
    db: Any,  # 仅为兼容签名，未使用
    col: Any,  # 仅为兼容签名，未使用
    query_dense_embedding: List[float],
    query_sparse_embedding: Any,  # 未用；保留签名
    limit: int,
    sparse_weight: float,
    dense_weight: float,
    expr: str,
    output_fields: Optional[List[str]] = None,
):
    """
    简化版本的 hybrid_search：目前仅对 dense_vector 做检索，
    但保留原函数签名，便于后续扩展为真正的稀疏+稠密混合。
    """
    adapter = MilvusAdapter()
    return adapter.search(
        collection_name=str(col) if isinstance(col, str) else "projects_documents_chunks_v2",
        query_vec=query_dense_embedding,
        limit=limit,
        expr=expr or None,
        output_fields=output_fields,
        anns_field="dense_vector",
        metric_type="IP",
    )
