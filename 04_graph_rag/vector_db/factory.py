"""向量数据库工厂类"""

import logging
from typing import Dict, Any, Optional

from .base import VectorDBInterface
from .milvus_adapter import MilvusAdapter

logger = logging.getLogger(__name__)


class VectorDBFactory:
    """向量数据库工厂类
    
    支持动态创建不同类型的向量数据库适配器。
    """

    _adapters = {
        "milvus": MilvusAdapter,
        # 未来可以添加其他适配器
        # "chroma": ChromaAdapter,
        # "faiss": FAISSAdapter,
    }

    @classmethod
    def create(
        cls,
        db_type: str,
        config: Optional[Dict[str, Any]] = None,
        milvus_manager=None,
    ) -> VectorDBInterface:
        """创建向量数据库适配器实例
        
        Args:
            db_type: 数据库类型，支持 "milvus", "chroma", "faiss" 等
            config: 配置字典（可选）
            milvus_manager: MilvusManager实例（仅用于milvus类型，可选）
            
        Returns:
            VectorDBInterface: 向量数据库适配器实例
            
        Raises:
            ValueError: 如果db_type不支持
        """
        if db_type not in cls._adapters:
            raise ValueError(
                f"不支持的向量数据库类型: {db_type}。"
                f"支持的类型: {list(cls._adapters.keys())}"
            )

        adapter_class = cls._adapters[db_type]
        logger.info(f"创建 {db_type} 向量数据库适配器")

        # 根据不同类型使用不同的初始化方式
        if db_type == "milvus":
            # Milvus适配器可以直接接收MilvusManager实例
            adapter = adapter_class(milvus_manager=milvus_manager)
        else:
            # 其他适配器使用config初始化
            adapter = adapter_class(config=config)

        return adapter

    @classmethod
    def register_adapter(cls, db_type: str, adapter_class):
        """注册新的适配器类型
        
        Args:
            db_type: 数据库类型名称
            adapter_class: 适配器类（必须实现VectorDBInterface）
        """
        if not issubclass(adapter_class, VectorDBInterface):
            raise ValueError(f"适配器类必须实现 VectorDBInterface 接口")
        cls._adapters[db_type] = adapter_class
        logger.info(f"注册新的向量数据库适配器: {db_type}")

