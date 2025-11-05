"""关系数据库存储"""

import logging
import hashlib
from typing import Dict, List, Optional
from datetime import datetime

from app.db.session import get_sz_pm_db

logger = logging.getLogger(__name__)


class DBStorage:
    """关系数据库存储类
    
    负责实体和关系的关系数据库存储。
    """

    def __init__(self):
        """初始化数据库存储"""
        pass

    async def save_entity(self, entity_data: Dict):
        """保存实体到关系数据库"""
        try:
            async with get_sz_pm_db() as db:
                # 检查是否已存在（只用entity_name）
                check_sql = """
                    SELECT ENTITY_ID, DESCRIPTION FROM FAI_SZ.KG_ENTITIES 
                    WHERE ENTITY_NAME = ?
                """
                cursor = await db.execute(check_sql, [entity_data["entity_name"]])
                existing = cursor.fetchone()

                if existing:
                    # 合并描述（LightRAG方式）
                    existing_id, existing_desc = existing
                    new_desc = entity_data["description"] or ""

                    # 使用分隔符合并
                    if existing_desc and new_desc and existing_desc != new_desc:
                        merged_desc = f"{existing_desc};{new_desc}"
                    else:
                        merged_desc = existing_desc or new_desc

                    update_sql = """
                        UPDATE FAI_SZ.KG_ENTITIES 
                        SET DESCRIPTION = ?, ENTITY_TYPE = ?, DOCUMENT_ID = ?, UPDATED_AT = ? 
                        WHERE ENTITY_ID = ?
                    """
                    await db.execute(
                        update_sql,
                        [
                            merged_desc,
                            entity_data["entity_type"],
                            entity_data["document_id"],
                            self._get_current_timestamp(),
                            existing_id,
                        ],
                    )
                    logger.info(f"🔄 合并实体: {entity_data['entity_name']}")
                else:
                    # 新增实体 - 生成稳定ID
                    entity_id = self._generate_entity_id(entity_data["entity_name"])
                    insert_sql = """
                        INSERT INTO FAI_SZ.KG_ENTITIES 
                        (ENTITY_ID, ENTITY_NAME, ENTITY_TYPE, DESCRIPTION, DOCUMENT_ID, CREATED_AT, UPDATED_AT)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    current_time = self._get_current_timestamp()
                    await db.execute(
                        insert_sql,
                        [
                            entity_id,
                            entity_data["entity_name"],
                            entity_data["entity_type"],
                            entity_data["description"],
                            entity_data["document_id"],
                            current_time,
                            current_time,
                        ],
                    )
                    logger.info(
                        f"➕ 新增实体: {entity_data['entity_name']} (ID: {entity_id})"
                    )

                await db.commit()

        except Exception as e:
            logger.error(f"保存实体失败: {str(e)}")
            raise

    async def save_relation(self, relation_data: Dict):
        """保存关系到关系数据库"""
        try:
            async with get_sz_pm_db() as db:
                # 检查是否已存在（基于实体对和类型）
                check_sql = """
                    SELECT RELATION_ID, DESCRIPTION, KEYWORDS, WEIGHT FROM FAI_SZ.KG_RELATIONS 
                    WHERE SOURCE_ENTITY = ? AND TARGET_ENTITY = ? AND RELATION_TYPE = ?
                """
                cursor = await db.execute(
                    check_sql,
                    [
                        relation_data["source_entity"],
                        relation_data["target_entity"],
                        relation_data["relation_type"],
                    ],
                )
                existing = cursor.fetchone()

                if existing:
                    # 合并关系信息
                    existing_id, existing_desc, existing_keywords, existing_weight = (
                        existing
                    )

                    # 合并描述
                    new_desc = relation_data["description"] or ""
                    if existing_desc and new_desc and existing_desc != new_desc:
                        merged_desc = f"{existing_desc};{new_desc}"
                    else:
                        merged_desc = existing_desc or new_desc

                    # 合并关键词
                    existing_kw = existing_keywords or ""
                    new_kw = relation_data.get("keywords", "") or ""
                    merged_keywords = self._merge_keywords(existing_kw, new_kw)

                    # 权重相加
                    merged_weight = (existing_weight or 1.0) + (
                        relation_data.get("weight", 1.0)
                    )

                    update_sql = """
                        UPDATE FAI_SZ.KG_RELATIONS 
                        SET DESCRIPTION = ?, KEYWORDS = ?, WEIGHT = ?, DOCUMENT_ID = ?, UPDATED_AT = ? 
                        WHERE RELATION_ID = ?
                    """
                    await db.execute(
                        update_sql,
                        [
                            merged_desc,
                            merged_keywords,
                            merged_weight,
                            relation_data["document_id"],
                            self._get_current_timestamp(),
                            existing_id,
                        ],
                    )
                    logger.info(
                        f"🔄 合并关系: {relation_data['source_entity']} -> {relation_data['target_entity']}"
                    )
                else:
                    # 新增关系 - 生成稳定ID
                    relation_id = self._generate_relation_id(
                        relation_data["source_entity"],
                        relation_data["target_entity"],
                        relation_data["relation_type"],
                    )
                    insert_sql = """
                        INSERT INTO FAI_SZ.KG_RELATIONS 
                        (RELATION_ID, SOURCE_ENTITY, TARGET_ENTITY, RELATION_TYPE, DESCRIPTION, KEYWORDS, WEIGHT, DOCUMENT_ID, CREATED_AT, UPDATED_AT)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    current_time = self._get_current_timestamp()
                    await db.execute(
                        insert_sql,
                        [
                            relation_id,
                            relation_data["source_entity"],
                            relation_data["target_entity"],
                            relation_data["relation_type"],
                            relation_data["description"],
                            relation_data.get("keywords", ""),
                            relation_data.get("weight", 1.0),
                            relation_data["document_id"],
                            current_time,
                            current_time,
                        ],
                    )
                    logger.info(
                        f"➕ 新增关系: {relation_data['source_entity']} -> {relation_data['target_entity']} (ID: {relation_id})"
                    )

                await db.commit()

        except Exception as e:
            logger.error(f"保存关系失败: {str(e)}")
            raise

    async def init_kg_metadata(self, document_id: int):
        """初始化知识图谱元数据"""
        try:
            async with get_sz_pm_db() as db:
                # 检查是否已存在
                check_sql = (
                    "SELECT COUNT(*) FROM FAI_SZ.KG_METADATA WHERE DOCUMENT_ID = ?"
                )
                cursor = await db.execute(check_sql, [document_id])
                result = cursor.fetchone()
                exists = result[0] > 0 if result else False

                if not exists:
                    # 创建新的元数据记录
                    insert_sql = """
                        INSERT INTO FAI_SZ.KG_METADATA (DOCUMENT_ID, NAMESPACE, STATUS, ENTITY_COUNT, RELATION_COUNT, CREATED_AT, UPDATED_AT)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    current_time = self._get_current_timestamp()
                    await db.execute(
                        insert_sql,
                        [
                            document_id,
                            f"doc_{document_id}",
                            "processing",
                            0,
                            0,
                            current_time,
                            current_time,
                        ],
                    )
                    await db.commit()

        except Exception as e:
            logger.error(f"初始化元数据时出错: {str(e)}")
            raise

    async def update_kg_metadata(
        self, document_id: int, entity_count: int, relation_count: int, status: str
    ):
        """更新知识图谱元数据"""
        try:
            async with get_sz_pm_db() as db:
                update_sql = """
                    UPDATE FAI_SZ.KG_METADATA 
                    SET ENTITY_COUNT = ?, RELATION_COUNT = ?, STATUS = ?, UPDATED_AT = ?
                    WHERE DOCUMENT_ID = ?
                """
                current_time = self._get_current_timestamp()
                cursor = await db.execute(
                    update_sql,
                    [entity_count, relation_count, status, current_time, document_id],
                )
                await db.commit()

                # 检查是否更新成功
                if cursor.rowcount > 0:
                    logger.info(
                        f"更新文档 {document_id} 的元数据: 实体={entity_count}, 关系={relation_count}, 状态={status}"
                    )
                else:
                    logger.warning(f"文档 {document_id} 的元数据记录不存在，无法更新")

        except Exception as e:
            logger.error(f"更新元数据时出错: {str(e)}")
            raise

    async def query_entities_by_document(self, document_id: int) -> List[Dict]:
        """查询文档的所有实体"""
        try:
            async with get_sz_pm_db() as db:
                entity_sql = """
                    SELECT ENTITY_ID, ENTITY_NAME, ENTITY_TYPE, DESCRIPTION, DOCUMENT_ID, CREATED_AT, UPDATED_AT 
                    FROM FAI_SZ.KG_ENTITIES 
                    WHERE DOCUMENT_ID = ?
                """
                cursor = await db.execute(entity_sql, [document_id])
                entity_records = cursor.fetchall()

                result = []
                for record in entity_records:
                    result.append(
                        {
                            "entity_id": record[0],
                            "entity_name": record[1],
                            "entity_type": record[2],
                            "description": record[3],
                            "document_id": record[4],
                            "created_at": record[5],
                            "updated_at": record[6],
                        }
                    )

                logger.info(f"查询到文档 {document_id} 的 {len(result)} 个实体")
                return result

        except Exception as e:
            logger.error(f"查询实体时出错: {str(e)}")
            return []

    async def query_relations_by_document(self, document_id: int) -> List[Dict]:
        """查询文档的所有关系"""
        try:
            async with get_sz_pm_db() as db:
                relation_sql = """
                    SELECT RELATION_ID, SOURCE_ENTITY, TARGET_ENTITY, RELATION_TYPE, DESCRIPTION, KEYWORDS, WEIGHT, DOCUMENT_ID, CREATED_AT, UPDATED_AT 
                    FROM FAI_SZ.KG_RELATIONS 
                    WHERE DOCUMENT_ID = ?
                """
                cursor = await db.execute(relation_sql, [document_id])
                relation_records = cursor.fetchall()

                result = []
                for record in relation_records:
                    result.append(
                        {
                            "relation_id": record[0],
                            "source_entity": record[1],
                            "target_entity": record[2],
                            "relation_type": record[3],
                            "description": record[4],
                            "keywords": record[5],
                            "weight": record[6],
                            "document_id": record[7],
                            "created_at": record[8],
                            "updated_at": record[9],
                        }
                    )

                logger.info(f"查询到文档 {document_id} 的 {len(result)} 个关系")
                return result

        except Exception as e:
            logger.error(f"查询关系时出错: {str(e)}")
            return []

    def _generate_entity_id(self, entity_name: str) -> str:
        """生成稳定的实体ID"""
        return f"ent-{hashlib.md5(entity_name.encode('utf-8')).hexdigest()[:16]}"

    def _generate_relation_id(
        self, source: str, target: str, relation_type: str = "RELATED_TO"
    ) -> str:
        """生成稳定的关系ID"""
        content = f"{source}|{target}|{relation_type}"
        return f"rel-{hashlib.md5(content.encode('utf-8')).hexdigest()[:16]}"

    def _merge_keywords(self, existing_kw: str, new_kw: str) -> str:
        """关键词合并 - 去重排序"""
        if not existing_kw:
            return new_kw
        if not new_kw:
            return existing_kw

        # 去重合并
        all_keywords = set()
        for kw in (existing_kw + "," + new_kw).split(","):
            kw = kw.strip()
            if kw:
                all_keywords.add(kw)

        return ",".join(sorted(all_keywords))

    def _get_current_timestamp(self) -> str:
        """获取当前时间的TIMESTAMP格式字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

