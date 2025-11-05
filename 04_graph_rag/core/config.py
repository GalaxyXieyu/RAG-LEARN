"""配置管理"""

from typing import Optional, Dict, Any
from app.core.config.config import LlmClientType, MilvusDBName, MilvusCollectionName


class GraphRAGConfig:
    """GraphRAG配置类"""

    def __init__(
        self,
        vector_db_type: str = "milvus",
        vector_db_config: Optional[Dict[str, Any]] = None,
        model_type: Optional[LlmClientType] = None,
        db_name: str = MilvusDBName.SZAI,
        entity_collection: str = MilvusCollectionName.KG_ENTITIES,
        relation_collection: str = MilvusCollectionName.KG_RELATIONS,
        chunk_collection: str = MilvusCollectionName.KG_CHUNKS,
        graph_storage_dir: str = "storage/kg_graphs",
        prompt_file_path: Optional[str] = None,
    ):
        """初始化配置
        
        Args:
            vector_db_type: 向量数据库类型 ("milvus", "chroma", "faiss"等)
            vector_db_config: 向量数据库配置字典
            model_type: LLM客户端类型
            db_name: 数据库名称
            entity_collection: 实体集合名称
            relation_collection: 关系集合名称
            chunk_collection: 文档块集合名称
            graph_storage_dir: 图谱存储目录
            prompt_file_path: 提示词文件路径（可选，用于自定义业务提示词）
        """
        self.vector_db_type = vector_db_type
        self.vector_db_config = vector_db_config or {}
        self.model_type = model_type or LlmClientType.DeepSeekV3
        self.db_name = db_name
        self.entity_collection = entity_collection
        self.relation_collection = relation_collection
        self.chunk_collection = chunk_collection
        self.graph_storage_dir = graph_storage_dir
        self.prompt_file_path = prompt_file_path

