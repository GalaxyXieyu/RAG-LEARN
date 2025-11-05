"""向量数据库抽象接口"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorDBInterface(ABC):
    """向量数据库抽象接口
    
    所有向量数据库适配器都需要实现此接口，以支持不同的向量库实现。
    """

    @abstractmethod
    def connect(self) -> bool:
        """连接到向量数据库
        
        Returns:
            bool: 连接是否成功
        """
        pass

    @abstractmethod
    def get_collection(self, db_name: str, collection_name: str):
        """获取集合对象
        
        Args:
            db_name: 数据库名称
            collection_name: 集合名称
            
        Returns:
            集合对象，如果不存在则返回None
        """
        pass

    @abstractmethod
    def create_collection(
        self,
        db_name: str,
        collection_name: str,
        schema: Dict[str, Any],
        description: Optional[str] = None,
    ) -> bool:
        """创建集合
        
        Args:
            db_name: 数据库名称
            collection_name: 集合名称
            schema: 集合schema定义
            description: 集合描述
            
        Returns:
            bool: 创建是否成功
        """
        pass

    @abstractmethod
    def insert(
        self,
        db_name: str,
        collection_name: str,
        data: List[Dict[str, Any]],
    ) -> bool:
        """插入数据到集合
        
        Args:
            db_name: 数据库名称
            collection_name: 集合名称
            data: 要插入的数据列表，每个元素是一个字典
            
        Returns:
            bool: 插入是否成功
        """
        pass

    @abstractmethod
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
        """向量搜索
        
        Args:
            db_name: 数据库名称
            collection_name: 集合名称
            query_vector: 查询向量
            top_k: 返回最相似的top_k个结果
            expr: 过滤表达式（可选）
            output_fields: 返回的字段列表（可选）
            search_params: 搜索参数（可选）
            
        Returns:
            List[Dict]: 搜索结果列表，每个元素包含相似度和数据
        """
        pass

    @abstractmethod
    def delete(
        self,
        db_name: str,
        collection_name: str,
        ids: List[str],
    ) -> bool:
        """删除数据
        
        Args:
            db_name: 数据库名称
            collection_name: 集合名称
            ids: 要删除的ID列表
            
        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    def flush(self, db_name: str, collection_name: str) -> bool:
        """刷新集合，确保数据持久化
        
        Args:
            db_name: 数据库名称
            collection_name: 集合名称
            
        Returns:
            bool: 刷新是否成功
        """
        pass

