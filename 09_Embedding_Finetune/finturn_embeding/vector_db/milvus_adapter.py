"""
Milvus 适配器：统一连接、检索与插入接口

注意：为兼容现有集合 schema，列顺序与字段名沿用现有脚本：
['chunk_id','document_id','source_file','project_code','page_idx','block_type','stage','table_group_id','headers_norm','text','dense_vector', 'sparse_vector?']
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from pymilvus import connections, Collection


class MilvusAdapter:
    def __init__(self, host: str = "127.0.0.1", port: str = "19530", timeout: int = 10):
        connections.connect(host=host, port=port, timeout=timeout)

    def get_collection(self, name: str) -> Collection:
        col = Collection(name)
        col.load()
        return col

    def search(
        self,
        collection_name: str,
        query_vec: List[float],
        limit: int = 10,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        anns_field: str = "dense_vector",
        metric_type: str = "IP",
    ) -> List[Dict[str, Any]]:
        col = self.get_collection(collection_name)
        params = {"metric_type": metric_type, "params": {}}
        res = col.search(
            [query_vec],
            anns_field=anns_field,
            limit=limit,
            expr=expr,
            output_fields=output_fields or [],
            param=params,
        )[0]
        out: List[Dict[str, Any]] = []
        for hit in res:
            item = {"score": hit.score}
            if output_fields:
                for f in output_fields:
                    item[f] = hit.entity.get(f, "")
            out.append(item)
        return out

    def insert_rows(
        self,
        collection_name: str,
        rows: List[Dict[str, Any]],
        dense_vecs: List[List[float]],
        sparse_vecs: Optional[List[Any]] = None,
    ) -> int:
        """
        按既有 schema 列序插入，返回成功条数。
        
        注意：稀疏向量字段已弃用（qwen3-embedding-0.6b 不支持），
        sparse_vecs 参数保留仅为向后兼容。
        """
        col = self.get_collection(collection_name)
        col_data = [
            [r.get("chunk_id", "") for r in rows],
            [r.get("document_id", "") for r in rows],
            [r.get("source_file", "") for r in rows],
            [r.get("project_code", "") for r in rows],
            [int(r.get("page_idx", -1)) for r in rows],
            [r.get("block_type", "") for r in rows],
            [r.get("stage", "") for r in rows],
            [r.get("table_group_id", "") for r in rows],
            [r.get("headers_norm", "") for r in rows],
            [r.get("text", "") for r in rows],
            dense_vecs,
        ]
        # 稀疏向量已弃用，但为了兼容旧的 collection schema，
        # 如果 schema 中有 sparse_vector 字段，仍需传入
        # 实际使用时建议重建 collection 移除该字段
        if sparse_vecs is not None:
            col_data.append(sparse_vecs)
        mr = col.insert(col_data)
        return len(rows)
