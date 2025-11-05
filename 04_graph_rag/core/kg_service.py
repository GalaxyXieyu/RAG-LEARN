"""知识图谱服务 - 模块化版本"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

try:
    from ..vector_db.factory import VectorDBFactory
    from ..document_processor.file_importer import FileImporter
    from ..knowledge_extractor.knowledge_extractor import KnowledgeExtractor
    from ..storage.graph_storage import GraphStorage
    from ..storage.db_storage import DBStorage
    from ..storage.vector_storage import VectorStorage
    from ..query.graphrag_query import GraphRAGQueryEngine
    from ..query.keyword_extractor import KeywordExtractor
except ImportError:
    # 如果相对导入失败，使用绝对导入
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from vector_db.factory import VectorDBFactory
    from document_processor.file_importer import FileImporter
    from knowledge_extractor.knowledge_extractor import KnowledgeExtractor
    from storage.graph_storage import GraphStorage
    from storage.db_storage import DBStorage
    from storage.vector_storage import VectorStorage
    from query.graphrag_query import GraphRAGQueryEngine
    from query.keyword_extractor import KeywordExtractor
from .config import GraphRAGConfig
from app.core.config.config import MilvusDBName, MilvusCollectionName, LlmClientType
from app.embeddings.milvus_client import MilvusManager

logger = logging.getLogger(__name__)


class KGService:
    """知识图谱服务类 - 模块化版本
    
    使用依赖注入组合各个模块，提供高级API。
    """

    def __init__(
        self,
        config: Optional[GraphRAGConfig] = None,
        vector_db=None,
        document_processor=None,
        knowledge_extractor=None,
        graph_storage=None,
        db_storage=None,
        vector_storage=None,
        query_engine=None,
        model_type: Optional[LlmClientType] = None,
        milvus_manager=None,
    ):
        """初始化KGService
        
        Args:
            config: 配置对象
            vector_db: 向量数据库适配器（可选，会根据config创建）
            document_processor: 文档处理器（可选）
            knowledge_extractor: 知识抽取器（可选）
            graph_storage: 图谱存储（可选）
            db_storage: 数据库存储（可选）
            vector_storage: 向量存储（可选）
            query_engine: 查询引擎（可选）
            model_type: LLM客户端类型（向后兼容）
            milvus_manager: MilvusManager实例（向后兼容）
        """
        # 配置
        self.config = config or GraphRAGConfig()
        
        # 如果提供了model_type（向后兼容），更新config
        if model_type:
            self.config.model_type = model_type

        # 向量数据库
        if vector_db is None:
            if milvus_manager:
                # 向后兼容：使用提供的MilvusManager
                try:
                    from ..vector_db.milvus_adapter import MilvusAdapter
                except ImportError:
                    from vector_db.milvus_adapter import MilvusAdapter
                vector_db = MilvusAdapter(milvus_manager=milvus_manager)
            else:
                vector_db = VectorDBFactory.create(
                    self.config.vector_db_type,
                    self.config.vector_db_config,
                    milvus_manager=milvus_manager,
                )
        self.vector_db = vector_db

        # 向量存储
        self.vector_storage = vector_storage or VectorStorage(
            vector_db=self.vector_db
        )

        # 图谱存储
        self.graph_storage = graph_storage or GraphStorage(
            graph_storage_dir=self.config.graph_storage_dir
        )

        # 数据库存储
        self.db_storage = db_storage or DBStorage()

        # 文档处理器
        self.document_processor = document_processor or FileImporter()

        # 知识抽取器（支持提示词配置）
        if knowledge_extractor is None:
            # 检查config中是否有提示词配置
            prompt_file_path = getattr(self.config, 'prompt_file_path', None)
            if prompt_file_path:
                self.knowledge_extractor = KnowledgeExtractor(
                    model_type=self.config.model_type,
                    prompt_file_path=prompt_file_path
                )
            else:
                self.knowledge_extractor = KnowledgeExtractor(
                    model_type=self.config.model_type
                )
        else:
            self.knowledge_extractor = knowledge_extractor

        # 查询引擎
        self.query_engine = query_engine or GraphRAGQueryEngine(
            vector_storage=self.vector_storage,
            graph_storage=self.graph_storage,
            db_storage=self.db_storage,
            keyword_extractor=KeywordExtractor(model_type=self.config.model_type),
        )

        # 向后兼容：保留原有的属性
        self.milvus_manager = milvus_manager or self._get_milvus_manager()
        self.embedding_func = self.vector_storage.embedding_func
        self.model_type = self.config.model_type
        self.knowledge_graph = self.graph_storage.get_graph()

    def _get_milvus_manager(self):
        """获取MilvusManager实例（向后兼容）"""
        if isinstance(self.vector_db, type(self.vector_db)):
            # 如果是MilvusAdapter，尝试获取内部的MilvusManager
            if hasattr(self.vector_db, 'milvus_manager'):
                return self.vector_db.milvus_manager
        # 创建新的MilvusManager
        return MilvusManager()

    async def process_document_chunks(
        self, document_id: int, chunks: List[Dict[str, Any]], max_concurrent: int = 5
    ) -> Dict[str, Any]:
        """
        处理文档的所有chunks，抽取实体和关系并入库

        Args:
            document_id: 文档ID
            chunks: chunk列表，每个chunk包含 {"text": "...", "chunk_id": "...", "tokens": 123}
            max_concurrent: 最大并发LLM调用数

        Returns:
            处理结果统计
        """
        try:
            logger.info(
                f"开始处理文档 {document_id} 的 {len(chunks)} 个chunks，并发数: {max_concurrent}"
            )

            # 初始化元数据
            await self.db_storage.init_kg_metadata(document_id)

            # 并发抽取所有chunks的实体和关系
            all_entities, all_relations = (
                await self._batch_extract_from_chunks_optimized(
                    chunks, document_id, max_concurrent
                )
            )

            # 合并和入库
            entity_count = await self._merge_and_save_entities(
                document_id, all_entities
            )
            relation_count = await self._merge_and_save_relations(
                document_id, all_relations
            )

            # 保存chunks到向量库
            await self.vector_storage.batch_save_chunks(
                document_id,
                chunks,
                self.config.db_name,
                self.config.chunk_collection,
            )

            # 保存NetworkX图到文件
            self.graph_storage.save()

            await self.db_storage.update_kg_metadata(
                document_id, entity_count, relation_count, "ready"
            )

            result = {
                "document_id": document_id,
                "chunks_processed": len(chunks),
                "entities_extracted": entity_count,
                "relations_extracted": relation_count,
                "status": "success",
            }

            logger.info(f"文档 {document_id} 处理完成: {result}")
            return result

        except Exception as e:
            logger.error(f"处理文档 {document_id} 时出错: {str(e)}")
            await self.db_storage.update_kg_metadata(document_id, 0, 0, "error")
            raise

    async def _batch_extract_from_chunks_optimized(
        self, chunks: List[Dict[str, Any]], document_id: int, max_concurrent: int = 5
    ) -> Tuple[Dict, Dict]:
        """并发批量抽取chunks的实体和关系"""
        try:
            logger.info(
                f"🚀 优化并发抽取 {len(chunks)} 个chunks，并发数: {max_concurrent}"
            )

            semaphore = asyncio.Semaphore(max_concurrent)
            completed_chunks = 0
            completed_lock = asyncio.Lock()

            async def extract_with_semaphore(chunk, index):
                nonlocal completed_chunks
                async with semaphore:
                    chunk_id = chunk.get("chunk_id", self._generate_chunk_id(chunk.get("text", "")))
                    content = chunk.get("text", "")

                    try:
                        entities, relations = await self.knowledge_extractor.extract(
                            content, chunk_id, document_id
                        )

                        async with completed_lock:
                            completed_chunks += 1
                            progress = (completed_chunks / len(chunks)) * 100

                        print(
                            f"         ✅ 完成 {index+1}/{len(chunks)} ({progress:.1f}%): 实体={len(entities)}, 关系={len(relations)}"
                        )
                        return entities, relations

                    except Exception as e:
                        async with completed_lock:
                            completed_chunks += 1

                        print(
                            f"         ❌ 失败 {index+1}/{len(chunks)}: {str(e)[:50]}..."
                        )
                        logger.error(f"Chunk {chunk_id} 抽取失败: {e}")
                        return {}, {}

            # 创建所有并发任务
            tasks = [
                asyncio.create_task(extract_with_semaphore(chunk, i))
                for i, chunk in enumerate(chunks)
            ]

            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 合并所有结果
            all_entities = defaultdict(list)
            all_relations = defaultdict(list)

            successful_extractions = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Chunk {i+1} 处理异常: {result}")
                    continue

                entities, relations = result
                if entities or relations:
                    successful_extractions += 1

                # 实体按(名称, 类型)分组
                for entity_name, entity_data_list in entities.items():
                    for entity_data in entity_data_list:
                        entity_key = (
                            entity_data["entity_name"],
                            entity_data["entity_type"],
                        )
                        all_entities[entity_key].append(entity_data)

                # 关系按(源, 目标, 类型)分组
                for relation_key, relation_data_list in relations.items():
                    for relation_data in relation_data_list:
                        enhanced_key = (
                            relation_data["source_entity"],
                            relation_data["target_entity"],
                            relation_data["relation_type"],
                        )
                        all_relations[enhanced_key].append(relation_data)

            logger.info(
                f"✅ 优化并发抽取完成: {successful_extractions}/{len(chunks)} 个chunk成功"
            )

            return dict(all_entities), dict(all_relations)

        except Exception as e:
            logger.error(f"批量抽取失败: {str(e)}")
            raise

    async def _merge_and_save_entities(
        self, document_id: int, entities_dict: Dict[Tuple[str, str], List[Dict]]
    ) -> int:
        """合并并保存实体到数据库和向量库"""
        saved_count = 0
        merged_entities = []

        # 批量合并实体数据
        for entity_key, entity_data_list in entities_dict.items():
            try:
                # 简化合并逻辑（实际应该调用_merge_entity_data）
                merged_entity = entity_data_list[0].copy()
                # 合并描述
                descriptions = [e.get("description", "") for e in entity_data_list if e.get("description")]
                merged_entity["description"] = " | ".join(sorted(set(descriptions))) if descriptions else ""

                # 保存到关系数据库
                await self.db_storage.save_entity(merged_entity)

                # 保存到NetworkX图
                self.graph_storage.add_entity(merged_entity)

                merged_entities.append(merged_entity)
                saved_count += 1

            except Exception as e:
                logger.warning(f"跳过实体 '{entity_key}': {str(e)}")
                continue

        # 批量保存到向量库
        if merged_entities:
            await self.vector_storage.batch_save_entities(
                merged_entities,
                self.config.db_name,
                self.config.entity_collection,
            )

        logger.info(f"保存了 {saved_count} 个实体到数据库和向量库")
        return saved_count

    async def _merge_and_save_relations(
        self, document_id: int, relations_dict: Dict[Tuple, List[Dict]]
    ) -> int:
        """合并并保存关系到数据库和向量库"""
        saved_count = 0
        merged_relations = []

        # 批量合并关系数据
        for relation_data_list in relations_dict.values():
            try:
                # 简化合并逻辑
                merged_relation = relation_data_list[0].copy()
                # 合并描述和权重
                descriptions = [r.get("description", "") for r in relation_data_list if r.get("description")]
                merged_relation["description"] = " | ".join(sorted(set(descriptions))) if descriptions else ""
                merged_relation["weight"] = sum(r.get("weight", 1.0) for r in relation_data_list)

                # 保存到关系数据库
                await self.db_storage.save_relation(merged_relation)

                # 保存到NetworkX图
                self.graph_storage.add_relation(merged_relation)

                merged_relations.append(merged_relation)
                saved_count += 1

            except Exception as e:
                logger.warning(f"跳过关系: {str(e)}")
                continue

        # 批量保存到向量库
        if merged_relations:
            await self.vector_storage.batch_save_relations(
                merged_relations,
                self.config.db_name,
                self.config.relation_collection,
            )

        logger.info(f"保存了 {saved_count} 个关系到数据库和向量库")
        return saved_count

    async def graphrag_query(
        self, query: str, mode: str = "mix", top_k: int = 10
    ) -> Dict[str, Any]:
        """GraphRAG查询接口（向后兼容）"""
        return await self.query_engine.graphrag_query(query, mode, top_k)

    # 向后兼容方法
    async def extract_keywords_from_query(self, query: str) -> Tuple[List[str], List[str]]:
        """提取关键词（向后兼容）"""
        return await self.query_engine.keyword_extractor.extract_keywords(query)

    def _generate_chunk_id(self, content: str) -> str:
        """生成chunk ID"""
        import hashlib
        return f"chunk-{hashlib.md5(content.encode('utf-8')).hexdigest()[:16]}"

