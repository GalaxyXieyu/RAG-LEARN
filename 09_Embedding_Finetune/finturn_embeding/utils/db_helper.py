"""
数据库查询辅助函数
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.db.session import get_sz_project_db
from app.utils.param_sql import Where, build_select


async def fetch_report_files(
    project_code: Optional[str] = None,
    related_id: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    从数据库查询 report 类型的文件
    
    Args:
        project_code: 项目编码（可选）
        related_id: 任务ID（可选）
        limit: 限制返回数量（可选）
        
    Returns:
        文件记录列表，包含 id, origin_name, project_code, related_id, biz_type, object_key 等
    """
    try:
        async with get_sz_project_db() as session:
            where_clause = Where().eq("biz_type", "report").eq("is_deleted", 0)
            
            if project_code:
                where_clause = where_clause.eq("project_code", project_code)
            if related_id:
                where_clause = where_clause.eq("related_id", related_id)
            
            select_sql, params = build_select(
                table="tb_fai_file_storage_record",
                columns=["id", "origin_name", "project_code", "related_id", "biz_type", "object_key", "bucket_name"],
                where=where_clause,
                order_by="id DESC",
                limit=limit
            )
            
            cur = await session.execute(select_sql, params)
            rows = cur.fetchall() or []
            
            return [
                {
                    "id": row[0],
                    "origin_name": row[1],
                    "project_code": row[2],
                    "related_id": row[3],
                    "biz_type": row[4],
                    "object_key": row[5] if len(row) > 5 else None,
                    "bucket_name": row[6] if len(row) > 6 else None,
                }
                for row in rows
            ]
    except Exception as e:
        print(f"查询 report 文件失败: {e}")
        return []

