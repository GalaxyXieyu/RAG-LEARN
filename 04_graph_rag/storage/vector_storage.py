"""向量存储 - 使用向量数据库抽象层"""

import logging
import hashlib
from typing import List, Dict, Any
from datetime import datetime

from app.embeddings.BGEM3_embedding import NEWBGEM3EmbeddingFunction
from app.core.config.config import MilvusDBName, MilvusCollectionName
try:
    from ..vector_db.base import VectorDBInterface
except ImportError:
    from vector_db.base import VectorDBInterface

logger = logging.getLogger(__name__)


class VectorStorage:
    """向量存储类
    
    使用向量数据库抽象层存储实体、关系和文档块的向量。
    """

    def __init__(self, vector_db: VectorDBInterface, embedding_func=None):
        """初始化向量存储
        
        Args:
            vector_db: 向量数据库适配器实例
            embedding_func: Embedding函数，如果为None则使用默认的BGEM3
        """
        self.vector_db = vector_db
        self.embedding_func = embedding_func or NEWBGEM3EmbeddingFunction()

    async def batch_save_entities(self, entities: List[Dict], db_name: str, collection_name: str):
        """批量保存实体到向量数据库"""
        try:
            if not entities:
                return

            logger.info(f"开始批量保存 {len(entities)} 个实体向量")

            # 批量生成向量
            entity_texts = []
            insert_batch = []

            for entity_data in entities:
                # 生成实体的文本表示用于向量化
                entity_text = (
                    f"{entity_data['entity_name']}\n{entity_data['description']}"
                )
                entity_texts.append(entity_text)

            # 批量调用embedding
            if entity_texts:
                result = self.embedding_func(entity_texts)
                dense_vectors = result["dense"]

                # 准备批量插入数据
                for i, entity_data in enumerate(entities):
                    # 截断description字段以符合Milvus长度限制（4096字符）
                    description = entity_data["description"] or ""
                    if len(description) > 4096:
                        description = description[:4093] + "..."

                    # 截断entity_name字段以符合Milvus长度限制（256字符）
                    entity_name = entity_data["entity_name"] or ""
                    if len(entity_name) > 256:
                        entity_name = entity_name[:253] + "..."

                    insert_data = {
                        "entity_id": entity_data["entity_id"],
                        "entity_name": entity_name,
                        "entity_type": entity_data["entity_type"],
                        "description": description,
                        "document_id": str(entity_data["document_id"]),
                        "created_at": self._get_current_timestamp(),
                        "dense_vector": (
                            dense_vectors[i].tolist()
                            if hasattr(dense_vectors[i], "tolist")
                            else dense_vectors[i]
                        ),
                    }
                    insert_batch.append(insert_data)

                # 批量插入到向量库
                collection = self.vector_db.get_collection(db_name, collection_name)
                if collection:
                    collection.load()
                    collection.insert(insert_batch)
                    collection.flush()
                    logger.info(f"已批量插入 {len(insert_batch)} 个实体向量")

        except Exception as e:
            logger.error(f"批量保存实体向量时出错: {str(e)}")

    async def batch_save_relations(self, relations: List[Dict], db_name: str, collection_name: str):
        """批量保存关系到向量数据库"""
        try:
            if not relations:
                return

            logger.info(f"开始批量保存 {len(relations)} 个关系向量")

            # 批量生成向量
            relation_texts = []
            insert_batch = []

            for relation_data in relations:
                # 生成关系的文本表示用于向量化
                relation_text = f"{relation_data['keywords']}\t{relation_data['source_entity']}\n{relation_data['target_entity']}\n{relation_data['description']}"
                relation_texts.append(relation_text)

            # 批量调用embedding
            if relation_texts:
                result = self.embedding_func(relation_texts)
                dense_vectors = result["dense"]

                # 准备批量插入数据
                for i, relation_data in enumerate(relations):
                    # 截断description字段以符合Milvus长度限制（4096字符）
                    description = relation_data["description"] or ""
                    if len(description) > 4096:
                        description = description[:4093] + "..."

                    # 截断实体名称字段以符合Milvus长度限制（128字符）
                    source_entity = relation_data["source_entity"] or ""
                    if len(source_entity) > 128:
                        logger.warning(
                            f"截断source_entity: {len(source_entity)} -> 128"
                        )
                        source_entity = source_entity[:125] + "..."

                    target_entity = relation_data["target_entity"] or ""
                    if len(target_entity) > 128:
                        logger.warning(
                            f"截断target_entity: {len(target_entity)} -> 128"
                        )
                        target_entity = target_entity[:125] + "..."

                    insert_data = {
                        "relation_id": relation_data["relation_id"],
                        "source_entity": source_entity,
                        "target_entity": target_entity,
                        "relation_type": relation_data["relation_type"],
                        "description": description,
                        "keywords": relation_data["keywords"],
                        "weight": relation_data["weight"],
                        "document_id": str(relation_data["document_id"]),
                        "created_at": self._get_current_timestamp(),
                        "dense_vector": (
                            dense_vectors[i].tolist()
                            if hasattr(dense_vectors[i], "tolist")
                            else dense_vectors[i]
                        ),
                    }
                    insert_batch.append(insert_data)

                # 批量插入到向量库
                collection = self.vector_db.get_collection(db_name, collection_name)
                if collection:
                    collection.load()
                    collection.insert(insert_batch)
                    collection.flush()
                    logger.info(f"已批量插入 {len(insert_batch)} 个关系向量")

        except Exception as e:
            logger.error(f"批量保存关系向量时出错: {str(e)}")

    async def batch_save_chunks(self, document_id: int, chunks: List[Dict], db_name: str, collection_name: str):
        """批量保存chunks到向量数据库"""
        try:
            if not chunks:
                return

            logger.info(f"开始批量保存 {len(chunks)} 个chunk向量")

            # 批量生成向量
            chunk_texts = []
            insert_batch = []

            for chunk in chunks:
                chunk_texts.append(chunk.get("text", ""))

            # 批量调用embedding
            if chunk_texts:
                result = self.embedding_func(chunk_texts)
                dense_vectors = result["dense"]

                # 准备批量插入数据
                for i, chunk in enumerate(chunks):
                    chunk_id = chunk.get(
                        "chunk_id", self._generate_chunk_id(chunk.get("text", ""))
                    )

                    insert_data = {
                        "chunk_id": chunk_id,
                        "text": chunk.get("text", ""),
                        "document_id": str(document_id),
                        "chunk_order_index": chunk.get("chunk_order_index", 0),
                        "tokens": chunk.get("tokens", 0),
                        "page": chunk.get("page", 0),
                        "position": chunk.get("position", ""),
                        "source_file": chunk.get("source_file", ""),
                        "created_at": self._get_current_timestamp(),
                        "dense_vector": (
                            dense_vectors[i].tolist()
                            if hasattr(dense_vectors[i], "tolist")
                            else dense_vectors[i]
                        ),
                    }
                    insert_batch.append(insert_data)

                # 批量插入到向量库
                collection = self.vector_db.get_collection(db_name, collection_name)
                if collection:
                    collection.load()
                    collection.insert(insert_batch)
                    collection.flush()
                    logger.info(f"已批量插入 {len(insert_batch)} 个chunk向量")

        except Exception as e:
            logger.error(f"批量保存chunks向量时出错: {str(e)}")

    async def search_entities(
        self,
        query: str,
        db_name: str,
        collection_name: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """搜索实体向量"""
        try:
            # 获取查询的embedding
            query_embeddings = self.embedding_func([query])
            query_dense = query_embeddings["dense"][0].tolist()

            # 搜索向量库
            results = self.vector_db.search(
                db_name=db_name,
                collection_name=collection_name,
                query_vector=query_dense,
                top_k=top_k,
                output_fields=["entity_id", "entity_name", "entity_type", "description"],
            )

            entity_results = []
            for result in results:
                if result.get("score", 0.0) >= 0.05:  # 降低相似度阈值，增加召回率
                    entity_results.append({
                        "entity_id": result.get("entity_id"),
                        "entity_name": result.get("entity_name"),
                        "entity_type": result.get("entity_type"),
                        "description": result.get("description"),
                        "score": result.get("score", 0.0),
                    })

            return entity_results

        except Exception as e:
            logger.error(f"实体向量搜索失败: {str(e)}")
            return []

    async def search_relations(
        self,
        query: str,
        db_name: str,
        collection_name: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """搜索关系向量"""
        try:
            # 获取查询的embedding
            query_embeddings = self.embedding_func([query])
            query_dense = query_embeddings["dense"][0].tolist()

            # 搜索向量库
            results = self.vector_db.search(
                db_name=db_name,
                collection_name=collection_name,
                query_vector=query_dense,
                top_k=top_k,
                output_fields=[
                    "relation_id",
                    "source_entity",
                    "target_entity",
                    "relation_type",
                    "description",
                    "weight",
                ],
            )

            relation_results = []
            for result in results:
                if result.get("score", 0.0) >= 0.05:  # 降低相似度阈值，增加召回率
                    weight = result.get("weight")
                    if weight is None:
                        weight = 1.0
                    relation_results.append({
                        "relation_id": result.get("relation_id"),
                        "source_entity": result.get("source_entity"),
                        "target_entity": result.get("target_entity"),
                        "relation_type": result.get("relation_type"),
                        "description": result.get("description"),
                        "weight": weight,
                        "score": result.get("score", 0.0),
                    })

            return relation_results

        except Exception as e:
            logger.error(f"关系向量搜索失败: {str(e)}")
            return []

    async def search_chunks(
        self,
        query: str,
        db_name: str,
        collection_name: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """搜索文档块向量"""
        try:
            # 获取查询的embedding
            query_embeddings = self.embedding_func([query])
            query_dense = query_embeddings["dense"][0].tolist()

            # 搜索向量库
            results = self.vector_db.search(
                db_name=db_name,
                collection_name=collection_name,
                query_vector=query_dense,
                top_k=top_k,
                output_fields=[
                    "chunk_id",
                    "text",
                    "document_id",
                    "source_file",
                    "page",
                    "position",
                ],
            )

            chunk_results = []
            for result in results:
                if result.get("score", 0.0) >= 0.05:  # 降低相似度阈值，增加召回率
                    chunk_results.append({
                        "chunk_id": result.get("chunk_id"),
                        "text": result.get("text"),
                        "document_id": result.get("document_id"),
                        "score": result.get("score", 0.0),
                        "source_file": result.get("source_file"),
                        "page": result.get("page"),
                        "position": result.get("position"),
                    })

            return chunk_results

        except Exception as e:
            logger.error(f"文档块向量搜索失败: {str(e)}")
            return []

    def _generate_chunk_id(self, content: str) -> str:
        """生成chunk ID"""
        return f"chunk-{hashlib.md5(content.encode('utf-8')).hexdigest()[:16]}"

    def _get_current_timestamp(self) -> str:
        """获取当前时间的TIMESTAMP格式字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

