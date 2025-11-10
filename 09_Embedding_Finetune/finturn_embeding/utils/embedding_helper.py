"""
Embedding API 调用辅助函数
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.embeddings.BGEM3_embedding_api import BGEM3EmbeddingFunctionAPI
from finturn_embeding.vector_db.search import hybrid_search
from app.core.config.config import MilvusDBName, MilvusCollectionName
from finturn_embeding.config import EMBEDDING_API_CONFIG


def get_embedding_function():
    """
    获取 embedding 函数（API 模式）
    
    Returns:
        BGEM3EmbeddingFunctionAPI 实例
    """
    return BGEM3EmbeddingFunctionAPI(
        model_name=EMBEDDING_API_CONFIG["model_name"],
        base_url=EMBEDDING_API_CONFIG["base_url"],
        api_key=EMBEDDING_API_CONFIG["api_key"]
    )


def search_chunks_by_query(
    query: str,
    limit: int = 10,
    project_code: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    使用 embedding API 检索 chunks
    
    Args:
        query: 查询文本（因子名称）
        limit: 返回数量（默认10）
        project_code: 项目编码（可选，用于筛选）
        
    Returns:
        chunk 列表，每个包含 text, score 等字段
    """
    try:
        # === 使用 API 模式进行向量化 ===
        ef_api = get_embedding_function()
        query_embeddings = ef_api([query])
        
        # 构建筛选表达式
        expr = ""
        if project_code:
            expr = f'project_code == "{project_code}"'
        
        # === 使用混合检索 ===
        results = hybrid_search(
            db=MilvusDBName.SZAI,
            col=MilvusCollectionName.PROJECTS_DOCUMENTS_CHUNKS,
            query_dense_embedding=query_embeddings["dense"][0],
            query_sparse_embedding=query_embeddings["sparse"],
            limit=limit,
            sparse_weight=1.0,
            dense_weight=1.0,
            expr=expr,
            output_fields=["text", "project_code", "document_id", "page_idx"]
        )
        
        # === 处理返回结果 ===
        if not results:
            return []
        
        # 转换为字典列表
        formatted_results = []
        for hit in results:
            # 处理 Milvus Hit 对象
            text = getattr(hit, "text", "") or ""
            project_code = getattr(hit, "project_code", "") or ""
            document_id = getattr(hit, "document_id", "") or ""
            page_idx = getattr(hit, "page_idx", 0) or 0
            score = getattr(hit, "score", 0.0) or 0.0
            
            formatted_results.append({
                "text": text,
                "project_code": project_code,
                "document_id": document_id,
                "page_idx": page_idx,
                "score": score
            })
        
        return formatted_results
    except Exception as e:
        print(f"检索 chunks 失败: {e}")
        return []

