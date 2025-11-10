"""Chroma 向量数据库适配器"""

import logging
from typing import List, Dict, Any, Optional

from .base import VectorDBInterface

logger = logging.getLogger(__name__)


class ChromaAdapter(VectorDBInterface):
    """ChromaDB 向量数据库适配器
    
    说明：
    - 将 (db_name, collection_name) 组合映射为 Chroma 的单一 collection 名称
    - 写入时识别 `dense_vector` 字段为 embedding，其余字段作为 metadata
    - 读取时根据 output_fields 从 metadata 回填
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._client = None
        self._collections: Dict[str, Any] = {}

    def _compose_col_name(self, db_name: str, collection_name: str) -> str:
        return f"{db_name}__{collection_name}"

    def connect(self) -> bool:
        try:
            import chromadb
            from chromadb.config import Settings

            # 支持内存或持久化目录
            persist_dir = self.config.get("persist_directory") or self.config.get("persist_dir")
            if persist_dir:
                self._client = chromadb.Client(Settings(is_persistent=True, persist_directory=persist_dir))
            else:
                self._client = chromadb.Client(Settings())
            return True
        except Exception as e:
            logger.error(f"连接 Chroma 失败: {str(e)}")
            return False

    def _ensure_client(self):
        if self._client is None:
            ok = self.connect()
            if not ok:
                raise RuntimeError("Chroma 客户端未就绪")

    def get_collection(self, db_name: str, collection_name: str):
        try:
            self._ensure_client()
            col_key = self._compose_col_name(db_name, collection_name)
            if col_key in self._collections:
                return self._collections[col_key]

            # 如果不存在则创建（Chroma get_or_create 语义）
            collection = self._client.get_or_create_collection(name=col_key)
            self._collections[col_key] = collection
            return collection
        except Exception as e:
            logger.error(f"获取 Chroma 集合失败: {str(e)}")
            return None

    def create_collection(
        self,
        db_name: str,
        collection_name: str,
        schema: Dict[str, Any],
        description: Optional[str] = None,
    ) -> bool:
        try:
            self._ensure_client()
            col_key = self._compose_col_name(db_name, collection_name)
            self._client.get_or_create_collection(name=col_key)
            return True
        except Exception as e:
            logger.error(f"创建 Chroma 集合失败: {str(e)}")
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
            collection = self.get_collection(db_name, collection_name)
            if collection is None:
                return False

            ids: List[str] = []
            embeddings: List[List[float]] = []
            metadatas: List[Dict[str, Any]] = []

            for row in data:
                # 选择一个稳定 id 字段
                row_id = (
                    row.get("entity_id")
                    or row.get("relation_id")
                    or row.get("chunk_id")
                    or row.get("id")
                )
                if row_id is None:
                    # 兜底：使用内容哈希
                    import hashlib
                    seed = str(row.get("text") or row.get("description") or "")
                    row_id = hashlib.md5(seed.encode("utf-8")).hexdigest()

                vec = row.get("dense_vector")
                if vec is None:
                    # 忽略没有向量的数据
                    continue

                ids.append(str(row_id))
                embeddings.append(vec)

                meta = {k: v for k, v in row.items() if k not in ("dense_vector",)}
                metadatas.append(meta)

            if not ids:
                return True

            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
            return True
        except Exception as e:
            logger.error(f"Chroma 插入失败: {str(e)}")
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
            collection = self.get_collection(db_name, collection_name)
            if collection is None:
                return []

            where = None
            if search_params and isinstance(search_params.get("where"), dict):
                where = search_params.get("where")

            res = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["distances", "metadatas", "ids"],
            )

            results: List[Dict[str, Any]] = []
            ids = res.get("ids", [[]])[0] if res.get("ids") else []
            dists = res.get("distances", [[]])[0] if res.get("distances") else []
            metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []

            for i in range(min(len(ids), len(dists), len(metas))):
                item = {"id": ids[i], "score": float(dists[i])}
                meta = metas[i] or {}
                if output_fields:
                    for f in output_fields:
                        item[f] = meta.get(f)
                else:
                    # 默认返回所有 metadata
                    item.update(meta)
                results.append(item)

            return results
        except Exception as e:
            logger.error(f"Chroma 搜索失败: {str(e)}")
            return []

    def delete(
        self,
        db_name: str,
        collection_name: str,
        ids: List[str],
    ) -> bool:
        try:
            collection = self.get_collection(db_name, collection_name)
            if collection is None:
                return False
            if not ids:
                return True
            collection.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Chroma 删除失败: {str(e)}")
            return False

    def flush(self, db_name: str, collection_name: str) -> bool:
        # Chroma 基于 SQLite/duckdb 时会自动持久化，这里作为 no-op
        return True


