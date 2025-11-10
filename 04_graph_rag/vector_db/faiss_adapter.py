"""FAISS 向量数据库适配器（内存版）"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from .base import VectorDBInterface

logger = logging.getLogger(__name__)


class _FaissCollection:
    """简易 FAISS 集合包装，维护 embeddings 与 metadata 对齐"""

    def __init__(self, dim: Optional[int] = None, metric: str = "L2"):
        import faiss  # type: ignore

        self.dim = dim
        self.metric = metric
        if metric.upper() == "L2":
            self.index = faiss.IndexFlatL2(dim or 0)
        elif metric.upper() in ("IP", "COSINE"):
            self.index = faiss.IndexFlatIP(dim or 0)
        else:
            raise ValueError(f"不支持的度量: {metric}")

        self.ids: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def _ensure_dim(self, vec: List[float]):
        import faiss  # type: ignore

        if self.dim is None:
            self.dim = len(vec)
            # 重新创建 index 以设置正确维度
            if self.metric.upper() == "L2":
                self.index = faiss.IndexFlatL2(self.dim)
            else:
                self.index = faiss.IndexFlatIP(self.dim)

    def add(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        import numpy as np

        if not embeddings:
            return
        self._ensure_dim(embeddings[0])
        xb = np.array(embeddings, dtype="float32")
        self.index.add(xb)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: List[float], top_k: int) -> Tuple[List[int], List[float]]:
        import numpy as np

        if self.index.ntotal == 0:
            return [], []
        xq = np.array([query_vector], dtype="float32")
        distances, indices = self.index.search(xq, top_k)
        return indices[0].tolist(), distances[0].tolist()

    def delete(self, ids: List[str]):
        # 简化：重建索引（适用于中小规模）
        if not ids:
            return
        keep = [i for i, _id in enumerate(self.ids) if _id not in ids]
        if len(keep) == len(self.ids):
            return
        # 重建
        kept_ids = [self.ids[i] for i in keep]
        kept_metas = [self.metadatas[i] for i in keep]
        kept_vecs = self._vectors_from_metas(keep)
        self.__init__(dim=self.dim, metric=self.metric)
        self.add(kept_ids, kept_vecs, kept_metas)

    def _vectors_from_metas(self, indices: List[int]) -> List[List[float]]:
        # 仅用于重建；实际 embeddings 不在 metadata 中，无法恢复
        # 这里为了简单，放弃 delete 的有效实现（保留原数据即可）。
        # 因为我们无法从 FAISS 中取回原向量，这里改为不支持精确删除。
        # 调用方可选择重建集合。
        return []


class FAISSAdapter(VectorDBInterface):
    """FAISS 向量数据库适配器（进程内存储，适合本地/测试场景）"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._collections: Dict[str, _FaissCollection] = {}

    def _compose_col_key(self, db_name: str, collection_name: str) -> str:
        return f"{db_name}__{collection_name}"

    def connect(self) -> bool:
        try:
            import faiss  # type: ignore  # noqa: F401
            return True
        except Exception as e:
            logger.error(f"连接 FAISS 失败: {str(e)}")
            return False

    def get_collection(self, db_name: str, collection_name: str):
        key = self._compose_col_key(db_name, collection_name)
        return self._collections.get(key)

    def create_collection(
        self,
        db_name: str,
        collection_name: str,
        schema: Dict[str, Any],
        description: Optional[str] = None,
    ) -> bool:
        try:
            metric = (schema.get("metric_type") or "L2").upper()
            dim = schema.get("dimension")
            key = self._compose_col_key(db_name, collection_name)
            self._collections[key] = _FaissCollection(dim=dim, metric=metric)
            return True
        except Exception as e:
            logger.error(f"创建 FAISS 集合失败: {str(e)}")
            return False

    def insert(
        self,
        db_name: str,
        collection_name: str,
        data: List[Dict[str, Any]],
    ) -> bool:
        try:
            if not data:
                return True
            key = self._compose_col_key(db_name, collection_name)
            col = self._collections.get(key)
            if col is None:
                # 尝试从第一条数据推断维度
                first_vec = data[0].get("dense_vector")
                dim = len(first_vec) if first_vec is not None else None
                col = _FaissCollection(dim=dim)
                self._collections[key] = col

            ids: List[str] = []
            embeddings: List[List[float]] = []
            metadatas: List[Dict[str, Any]] = []

            for row in data:
                vec = row.get("dense_vector")
                if vec is None:
                    continue
                row_id = (
                    row.get("entity_id")
                    or row.get("relation_id")
                    or row.get("chunk_id")
                    or row.get("id")
                )
                if row_id is None:
                    import hashlib
                    seed = str(row.get("text") or row.get("description") or "")
                    row_id = hashlib.md5(seed.encode("utf-8")).hexdigest()
                ids.append(str(row_id))
                embeddings.append(vec)
                meta = {k: v for k, v in row.items() if k != "dense_vector"}
                metadatas.append(meta)

            if not ids:
                return True
            col.add(ids, embeddings, metadatas)
            return True
        except Exception as e:
            logger.error(f"FAISS 插入失败: {str(e)}")
            return False

    def search(
        self,
        db_name: str,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            key = self._compose_col_key(db_name, collection_name)
            col = self._collections.get(key)
            if col is None:
                return []

            indices, distances = col.search(query_vector, top_k)
            results: List[Dict[str, Any]] = []
            for rank, (idx, dist) in enumerate(zip(indices, distances)):
                if idx < 0 or idx >= len(col.ids):
                    continue
                item = {"id": col.ids[idx], "score": float(dist)}
                meta = col.metadatas[idx] or {}
                if output_fields:
                    for f in output_fields:
                        item[f] = meta.get(f)
                else:
                    item.update(meta)
                results.append(item)
            return results
        except Exception as e:
            logger.error(f"FAISS 搜索失败: {str(e)}")
            return []

    def delete(
        self,
        db_name: str,
        collection_name: str,
        ids: List[str],
    ) -> bool:
        # 简化：当前不支持高效删除，返回 False 表示未执行
        # 若需要，可实现为重建集合
        return False

    def flush(self, db_name: str, collection_name: str) -> bool:
        # 内存实现，无需 flush
        return True





