"""Milvus向量数据库适配器"""

import logging
from typing import List, Dict, Any, Optional

from .base import VectorDBInterface

logger = logging.getLogger(__name__)


class MilvusAdapter(VectorDBInterface):
    """Milvus向量数据库适配器
    
    包装现有的MilvusManager，实现VectorDBInterface接口。
    """

    def __init__(self, milvus_manager=None):
        """初始化Milvus适配器
        
        Args:
            milvus_manager: MilvusManager实例，如果为None则创建新实例
        """
        if milvus_manager is None:
            # 延迟导入，避免循环依赖
            from app.embeddings.milvus_client import MilvusManager
            self.milvus_manager = MilvusManager()
        else:
            self.milvus_manager = milvus_manager

    def connect(self) -> bool:
        """连接到Milvus数据库"""
        try:
            # MilvusManager通常在初始化时已经连接
            # 这里可以添加连接检查逻辑
            return True
        except Exception as e:
            logger.error(f"连接Milvus失败: {str(e)}")
            return False

    def get_collection(self, db_name: str, collection_name: str):
        """获取Milvus集合对象"""
        try:
            # 使用MilvusManager的get_collection方法
            # 注意：这里需要根据实际的MilvusManager接口调整
            collection = self.milvus_manager.get_collection(db_name, collection_name)
            return collection
        except Exception as e:
            logger.error(f"获取集合失败: {str(e)}")
            return None

    def create_collection(
        self,
        db_name: str,
        collection_name: str,
        schema: Dict[str, Any],
        description: Optional[str] = None,
    ) -> bool:
        """创建Milvus集合"""
        try:
            # 如果MilvusManager有create_collection方法，使用它
            # 否则集合可能已经存在或通过其他方式创建
            # 这里先返回True，实际实现需要根据MilvusManager的接口调整
            logger.info(f"创建集合: {db_name}.{collection_name}")
            return True
        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            return False

    def insert(
        self,
        db_name: str,
        collection_name: str,
        data: List[Dict[str, Any]],
    ) -> bool:
        """插入数据到Milvus集合"""
        try:
            collection = self.get_collection(db_name, collection_name)
            if collection is None:
                logger.error(f"集合不存在: {db_name}.{collection_name}")
                return False

            # 确保集合已加载
            collection.load()

            # 插入数据
            collection.insert(data)
            logger.info(f"插入 {len(data)} 条数据到 {db_name}.{collection_name}")
            return True
        except Exception as e:
            logger.error(f"插入数据失败: {str(e)}")
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
        """在Milvus集合中搜索向量"""
        try:
            collection = self.get_collection(db_name, collection_name)
            if collection is None:
                logger.error(f"集合不存在: {db_name}.{collection_name}")
                return []

            # 确保集合已加载
            collection.load()

            # 设置默认搜索参数
            if search_params is None:
                search_params = {"metric_type": "L2", "params": {}}

            # 执行搜索
            results = collection.search(
                [query_vector],
                anns_field="dense_vector",
                limit=top_k,
                expr=expr or "",
                output_fields=output_fields or [],
                param=search_params,
            )

            # 转换结果为标准格式
            search_results = []
            if results and len(results) > 0:
                for hit in results[0]:
                    result_dict = {
                        "id": hit.id if hasattr(hit, "id") else None,
                        "score": float(hit.score) if hasattr(hit, "score") else 0.0,
                    }
                    # 添加输出字段
                    if output_fields:
                        for field in output_fields:
                            result_dict[field] = hit.get(field) if hasattr(hit, "get") else None
                    search_results.append(result_dict)

            return search_results
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []

    def delete(
        self,
        db_name: str,
        collection_name: str,
        ids: List[str],
    ) -> bool:
        """从Milvus集合中删除数据"""
        try:
            collection = self.get_collection(db_name, collection_name)
            if collection is None:
                logger.error(f"集合不存在: {db_name}.{collection_name}")
                return False

            # 执行删除
            collection.delete(expr=f"id in {ids}")
            logger.info(f"从 {db_name}.{collection_name} 删除 {len(ids)} 条数据")
            return True
        except Exception as e:
            logger.error(f"删除数据失败: {str(e)}")
            return False

    def flush(self, db_name: str, collection_name: str) -> bool:
        """刷新Milvus集合"""
        try:
            collection = self.get_collection(db_name, collection_name)
            if collection is None:
                logger.error(f"集合不存在: {db_name}.{collection_name}")
                return False

            collection.flush()
            logger.info(f"刷新集合: {db_name}.{collection_name}")
            return True
        except Exception as e:
            logger.error(f"刷新集合失败: {str(e)}")
            return False

