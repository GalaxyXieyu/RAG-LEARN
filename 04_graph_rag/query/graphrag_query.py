"""GraphRAG查询引擎"""

import time
import json
import logging
from typing import List, Dict, Any, Tuple

try:
    from .keyword_extractor import KeywordExtractor
    from ..storage.vector_storage import VectorStorage
    from ..storage.graph_storage import GraphStorage
    from ..storage.db_storage import DBStorage
except ImportError:
    from keyword_extractor import KeywordExtractor
    from storage.vector_storage import VectorStorage
    from storage.graph_storage import GraphStorage
    from storage.db_storage import DBStorage
from app.core.config.config import MilvusDBName, MilvusCollectionName

logger = logging.getLogger(__name__)


class GraphRAGQueryEngine:
    """GraphRAG查询引擎
    
    提供local_search, global_search, vector_search和graphrag_query方法。
    """

    def __init__(
        self,
        vector_storage: VectorStorage,
        graph_storage: GraphStorage,
        db_storage: DBStorage = None,
        keyword_extractor: KeywordExtractor = None,
    ):
        """初始化GraphRAG查询引擎
        
        Args:
            vector_storage: 向量存储实例
            graph_storage: 图谱存储实例
            db_storage: 数据库存储实例（可选，用于获取文档块）
            keyword_extractor: 关键词提取器实例
        """
        self.vector_storage = vector_storage
        self.graph_storage = graph_storage
        self.db_storage = db_storage or DBStorage()
        self.keyword_extractor = keyword_extractor or KeywordExtractor()

    async def local_search(
        self, low_level_keywords: List[str], top_k: int = 10
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Local检索：基于低层关键词从实体向量库检索，然后进行图遍历

        Args:
            low_level_keywords: 低层关键词列表
            top_k: 每个关键词检索的最大结果数

        Returns:
            (entities_context, relations_context, text_units_context)
        """
        try:
            entities_context = []
            relations_context = []
            text_units_context = []

            if not low_level_keywords:
                return entities_context, relations_context, text_units_context

            # 合并所有关键词进行向量检索
            query_text = " ".join(low_level_keywords)

            # 从实体向量库检索相关实体
            entity_results = await self.vector_storage.search_entities(
                query_text,
                MilvusDBName.SZAI,
                MilvusCollectionName.KG_ENTITIES,
                top_k,
            )

            # 收集相关实体ID
            relevant_entity_ids = set()
            for result in entity_results:
                entity_id = result.get("entity_id")
                if entity_id:
                    relevant_entity_ids.add(entity_id)

                    # 添加实体信息到上下文
                    entities_context.append({
                        "id": len(entities_context) + 1,
                        "entity_name": result.get("entity_name", ""),
                        "entity_type": result.get("entity_type", ""),
                        "description": result.get("description", ""),
                        "score": result.get("score", 0.0),
                    })

            # 基于相关实体进行图遍历，获取相关关系和邻居实体
            knowledge_graph = self.graph_storage.get_graph()
            logger.info(f"开始图遍历，相关实体ID: {list(relevant_entity_ids)}")

            for entity_id in relevant_entity_ids:
                if entity_id in knowledge_graph:
                    # 获取邻居节点和边
                    neighbors = list(knowledge_graph.neighbors(entity_id))
                    logger.debug(f"实体 {entity_id} 的邻居: {neighbors}")

                    # 添加邻居实体
                    for neighbor_id in neighbors:
                        if neighbor_id not in relevant_entity_ids:
                            neighbor_data = knowledge_graph.nodes[neighbor_id]
                            entities_context.append({
                                "id": len(entities_context) + 1,
                                "entity_name": neighbor_data.get("entity_name", ""),
                                "entity_type": neighbor_data.get("entity_type", ""),
                                "description": neighbor_data.get("description", ""),
                                "score": 0.5,  # 间接相关的权重较低
                            })

                    # 添加相关关系
                    for neighbor_id in neighbors:
                        if knowledge_graph.has_edge(entity_id, neighbor_id):
                            edge_data = knowledge_graph.edges[entity_id, neighbor_id]
                            relations_context.append({
                                "id": len(relations_context) + 1,
                                "source_entity": knowledge_graph.nodes[entity_id].get("entity_name", entity_id),
                                "target_entity": knowledge_graph.nodes[neighbor_id].get("entity_name", neighbor_id),
                                "relation_type": edge_data.get("relation_type", "RELATED_TO"),
                                "description": edge_data.get("description", ""),
                                "weight": edge_data.get("weight", 1.0),
                                "score": 0.6,  # 图遍历找到的关系给一个中等分数
                            })
                else:
                    logger.debug(f"实体 {entity_id} 不在图中")

            # 改进文档块检索：使用向量搜索
            chunk_results = await self.vector_storage.search_chunks(
                query_text,
                MilvusDBName.SZAI,
                MilvusCollectionName.KG_CHUNKS,
                top_k,
            )

            for i, result in enumerate(chunk_results):
                text_units_context.append({
                    "id": i + 1,
                    "text": result.get("text", ""),
                    "document_id": result.get("document_id", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "score": result.get("score", 0.0),
                    "source_file": result.get("source_file", ""),
                    "page": result.get("page", ""),
                    "position": result.get("position", ""),
                })

            logger.info(
                f"Local检索结果: {len(entities_context)}个实体, {len(relations_context)}个关系, {len(text_units_context)}个文档块"
            )
            return entities_context, relations_context, text_units_context

        except Exception as e:
            logger.error(f"Local检索失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], [], []

    async def global_search(
        self, high_level_keywords: List[str], top_k: int = 10
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Global检索：基于高层关键词从关系向量库检索，然后进行图遍历

        Args:
            high_level_keywords: 高层关键词列表
            top_k: 每个关键词检索的最大结果数

        Returns:
            (entities_context, relations_context, text_units_context)
        """
        try:
            entities_context = []
            relations_context = []
            text_units_context = []

            if not high_level_keywords:
                return entities_context, relations_context, text_units_context

            # 合并所有关键词进行向量检索
            query_text = " ".join(high_level_keywords)

            # 从关系向量库检索相关关系
            relation_results = await self.vector_storage.search_relations(
                query_text,
                MilvusDBName.SZAI,
                MilvusCollectionName.KG_RELATIONS,
                top_k,
            )

            # 收集相关实体ID
            relevant_entity_ids = set()
            for result in relation_results:
                source_entity = result.get("source_entity")
                target_entity = result.get("target_entity")

                if source_entity:
                    relevant_entity_ids.add(source_entity)
                if target_entity:
                    relevant_entity_ids.add(target_entity)

                # 添加关系信息到上下文
                relations_context.append({
                    "id": len(relations_context) + 1,
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "relation_type": result.get("relation_type", ""),
                    "description": result.get("description", ""),
                    "weight": result.get("weight", 1.0),
                    "score": result.get("score", 0.0),
                })

            # 添加相关实体信息
            knowledge_graph = self.graph_storage.get_graph()
            for entity_id in relevant_entity_ids:
                if entity_id in knowledge_graph:
                    entity_data = knowledge_graph.nodes[entity_id]
                    entities_context.append({
                        "id": len(entities_context) + 1,
                        "entity_name": entity_data.get("entity_name", ""),
                        "entity_type": entity_data.get("entity_type", ""),
                        "description": entity_data.get("description", ""),
                        "score": 0.8,  # 通过关系发现的实体权重较高
                    })

            # 从相关实体获取原始文档块
            text_units_context = await self._get_text_units_from_entities(
                list(relevant_entity_ids), top_k
            )

            logger.info(
                f"Global检索结果: {len(entities_context)}个实体, {len(relations_context)}个关系, {len(text_units_context)}个文档块"
            )
            return entities_context, relations_context, text_units_context

        except Exception as e:
            logger.error(f"Global检索失败: {str(e)}")
            return [], [], []

    async def vector_search(
        self, query: str, top_k: int = 10
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Vector检索：直接从文档块向量库检索相关内容

        Args:
            query: 原始查询文本
            top_k: 检索的最大结果数

        Returns:
            (entities_context, relations_context, text_units_context)
        """
        try:
            # Vector检索主要返回文档块，实体和关系为空
            entities_context = []
            relations_context = []
            text_units_context = []

            # 从文档块向量库检索
            chunk_results = await self.vector_storage.search_chunks(
                query,
                MilvusDBName.SZAI,
                MilvusCollectionName.KG_CHUNKS,
                top_k,
            )

            for i, result in enumerate(chunk_results):
                text_units_context.append({
                    "id": i + 1,
                    "text": result.get("text", ""),
                    "document_id": result.get("document_id", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "score": result.get("score", 0.0),
                    "source_file": result.get("source_file", ""),
                    "page": result.get("page", ""),
                    "position": result.get("position", ""),
                })

            logger.info(f"Vector检索结果: {len(text_units_context)}个文档块")
            return entities_context, relations_context, text_units_context

        except Exception as e:
            logger.error(f"Vector检索失败: {str(e)}")
            return [], [], []

    async def graphrag_query(
        self, query: str, mode: str = "mix", top_k: int = 10
    ) -> Dict[str, Any]:
        """
        GraphRAG查询接口，支持多种查询模式

        Args:
            query: 用户查询文本
            mode: 查询模式 ("local", "global", "hybrid", "mix")
            top_k: 检索的最大结果数

        Returns:
            包含实体、关系和文档块的结构化上下文
        """
        start_time = time.time()

        try:
            logger.info(f"开始GraphRAG查询: {query}, 模式: {mode}")

            # 1. 提取关键词
            high_level_keywords, low_level_keywords = (
                await self.keyword_extractor.extract_keywords(query)
            )

            # 2. 根据模式进行检索
            if mode == "local":
                # 只进行Local检索
                entities_context, relations_context, text_units_context = (
                    await self.local_search(low_level_keywords, top_k)
                )

            elif mode == "global":
                # 只进行Global检索
                entities_context, relations_context, text_units_context = (
                    await self.global_search(high_level_keywords, top_k)
                )

            elif mode == "hybrid":
                # 进行Local + Global检索
                ll_entities, ll_relations, ll_text_units = await self.local_search(
                    low_level_keywords, top_k
                )
                hl_entities, hl_relations, hl_text_units = await self.global_search(
                    high_level_keywords, top_k
                )

                # 合并结果
                entities_context = self._process_combine_contexts(
                    ll_entities, hl_entities
                )
                relations_context = self._process_combine_contexts(
                    ll_relations, hl_relations
                )
                text_units_context = self._process_combine_contexts(
                    ll_text_units, hl_text_units
                )

            elif mode == "mix":
                # 进行Local + Global + Vector检索
                ll_entities, ll_relations, ll_text_units = await self.local_search(
                    low_level_keywords, top_k
                )
                hl_entities, hl_relations, hl_text_units = await self.global_search(
                    high_level_keywords, top_k
                )
                vector_entities, vector_relations, vector_text_units = (
                    await self.vector_search(query, top_k)
                )

                # 合并结果
                entities_context = self._process_combine_contexts(
                    ll_entities, hl_entities, vector_entities
                )
                relations_context = self._process_combine_contexts(
                    ll_relations, hl_relations, vector_relations
                )
                text_units_context = self._process_combine_contexts(
                    ll_text_units, hl_text_units, vector_text_units
                )

            else:
                raise ValueError(f"不支持的查询模式: {mode}")

            # 3. 构建最终上下文
            processing_time = time.time() - start_time
            result = {
                "query": query,
                "mode": mode,
                "high_level_keywords": high_level_keywords,
                "low_level_keywords": low_level_keywords,
                "entities_count": len(entities_context),
                "relations_count": len(relations_context),
                "text_units_count": len(text_units_context),
                "entities": entities_context,
                "relations": relations_context,
                "text_units": text_units_context,
                "formatted_context": self._format_context_for_llm(
                    entities_context, relations_context, text_units_context
                ),
                "processing_time": processing_time,
            }

            logger.info(
                f"GraphRAG查询完成: {len(entities_context)}个实体, {len(relations_context)}个关系, {len(text_units_context)}个文档块, 耗时: {processing_time:.2f}秒"
            )
            return result

        except Exception as e:
            logger.error(f"GraphRAG查询失败: {str(e)}")
            processing_time = time.time() - start_time
            return {
                "query": query,
                "mode": mode,
                "error": str(e),
                "entities": [],
                "relations": [],
                "text_units": [],
                "formatted_context": "",
                "processing_time": processing_time,
            }

    def _process_combine_contexts(self, *context_lists: List[Dict]) -> List[Dict]:
        """合并并去重多个上下文列表"""
        seen_content = {}
        combined_data = []

        for context_list in context_lists:
            for item in context_list:
                # 创建内容唯一键（排除id字段）
                content_dict = {k: v for k, v in item.items() if k != "id"}
                content_key = tuple(sorted(content_dict.items()))

                # 去重处理
                if content_key not in seen_content:
                    seen_content[content_key] = item
                    combined_data.append(item)

        combined_data = sorted(combined_data, key=lambda x: x.get("score", 0.0), reverse=True)

        # 重新分配ID
        for i, item in enumerate(combined_data):
            item["id"] = str(i + 1)

        return combined_data

    def _format_context_for_llm(
        self, entities: List[Dict], relations: List[Dict], text_units: List[Dict]
    ) -> str:
        """格式化上下文为LLM可理解的格式"""
        # 转换为JSON格式
        entities_str = json.dumps(entities, ensure_ascii=False, indent=2)
        relations_str = json.dumps(relations, ensure_ascii=False, indent=2)
        text_units_str = json.dumps(text_units, ensure_ascii=False, indent=2)

        formatted_context = f"""-----实体信息(Entities)-----

        ```json
        {entities_str}
        ```

        -----关系信息(Relationships)-----

        ```json
        {relations_str}
        ```

        -----文档块(Document Chunks)-----

        ```json
        {text_units_str}
        ```
        """

        return formatted_context

    async def _get_text_units_from_entities(
        self, entity_ids: List[str], top_k: int = 10
    ) -> List[Dict]:
        """根据实体ID获取相关的文档块"""
        try:
            from app.db.session import get_sz_pm_db
            
            text_units = []

            # 从数据库查询实体对应的文档ID
            async with get_sz_pm_db() as db:
                if not entity_ids:
                    return []

                entity_ids_str = "', '".join(entity_ids)
                entity_sql = f"""
                    SELECT DISTINCT DOCUMENT_ID FROM FAI_SZ.KG_ENTITIES 
                    WHERE ENTITY_ID IN ('{entity_ids_str}') 
                    LIMIT {top_k}
                """
                cursor = await db.execute(entity_sql)
                entity_records = cursor.fetchall()

                document_ids = [record[0] for record in entity_records]

                if document_ids:
                    # 从文档块向量库检索这些文档的chunks
                    # 注意：document_id字段是VARCHAR类型，需要加引号
                    doc_ids_list = [f'"{str(doc_id)}"' for doc_id in set(document_ids)]
                    doc_ids_str = ",".join(doc_ids_list)
                    expr = f"document_id in [{doc_ids_str}]"

                    # 使用向量库搜索（带expr过滤）
                    # 这里需要使用dummy向量进行expr过滤查询
                    # 简化实现：直接搜索所有chunks，然后过滤
                    chunk_results = await self.vector_storage.search_chunks(
                        "",  # 空查询，主要用expr过滤
                        MilvusDBName.SZAI,
                        MilvusCollectionName.KG_CHUNKS,
                        top_k,
                    )

                    # 过滤document_id
                    filtered_chunks = [
                        chunk for chunk in chunk_results
                        if chunk.get("document_id") in [str(doc_id) for doc_id in document_ids]
                    ]

                    for i, result in enumerate(filtered_chunks[:top_k]):
                        text_units.append({
                            "id": i + 1,
                            "text": result.get("text", ""),
                            "document_id": result.get("document_id", ""),
                            "chunk_id": result.get("chunk_id", ""),
                            "source_file": result.get("source_file", ""),
                            "page": result.get("page", ""),
                            "position": result.get("position", ""),
                        })

            return text_units

        except Exception as e:
            logger.error(f"获取文档块失败: {str(e)}")
            return []

