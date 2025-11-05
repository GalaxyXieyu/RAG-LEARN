"""关系抽取器"""

import re
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from langchain_core.messages import HumanMessage
from app.llm.base_chat import chat_async
from app.core.config.config import LlmClientType

try:
    from .prompt_manager import PromptManager
except ImportError:
    from prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class RelationExtractor:
    """关系抽取器
    
    从文本中抽取关系信息。支持通过PromptManager灵活配置业务提示词。
    """

    def __init__(
        self,
        model_type: Optional[LlmClientType] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """初始化关系抽取器
        
        Args:
            model_type: LLM客户端类型
            prompt_manager: 提示词管理器实例（可选）
        """
        if model_type is None:
            model_type = LlmClientType.DeepSeekV3
        self.model_type = model_type
        
        # 使用提供的prompt_manager或创建默认的
        self.prompt_manager = prompt_manager or PromptManager()

    async def extract(
        self,
        content: str,
        chunk_id: str,
        document_id: int,
    ) -> Dict:
        """从文本中抽取关系
        
        Args:
            content: 文本内容
            chunk_id: chunk ID
            document_id: 文档ID
            
        Returns:
            Dict: 关系字典，key为(source_entity, target_entity)，value为关系数据列表
        """
        try:
            # 注意：在实际实现中，关系和实体可能在一个LLM调用中一起抽取
            # 这里为了模块化，我们假设关系抽取独立进行
            # 在实际使用中，KnowledgeExtractor会协调实体和关系的抽取
            
            # 这里返回空字典，实际的关系抽取逻辑在统一的抽取流程中
            # 可以参考原代码中的_parse_extraction_result方法
            return {}

        except Exception as e:
            logger.error(f"关系抽取失败 (chunk {chunk_id}): {str(e)}")
            return {}

    async def _parse_extraction_result(
        self,
        response: str,
        chunk_id: str,
        document_id: int,
    ) -> Dict:
        """解析LLM的关系抽取结果"""
        relations = defaultdict(list)

        if not response:
            return dict(relations)

        # 分割记录
        records = self._split_string_by_markers(
            response, [self.prompt_manager.record_delimiter, self.prompt_manager.completion_delimiter]
        )

        for record in records:
            record = record.strip()
            if not record:
                continue

            # 提取括号内的内容
            match = re.search(r"\((.*)\)", record)
            if not match:
                continue

            record_content = match.group(1)
            attributes = record_content.split(self.prompt_manager.tuple_delimiter)

            if len(attributes) < 5:
                continue

            # 处理关系
            if '"relationship"' in attributes[0]:
                relation_data = await self._handle_relation_extraction(
                    attributes, chunk_id, document_id
                )
                if relation_data:
                    relation_key = (
                        relation_data["source_entity"],
                        relation_data["target_entity"],
                    )
                    relations[relation_key].append(relation_data)

        return dict(relations)

    async def _handle_relation_extraction(
        self,
        attributes: List[str],
        chunk_id: str,
        document_id: int,
    ) -> Optional[Dict]:
        """处理单个关系抽取结果"""
        try:
            if len(attributes) < 5:
                return None

            source_entity = self._clean_string(attributes[1]).strip()
            target_entity = self._clean_string(attributes[2]).strip()
            description = self._clean_string(attributes[3]).strip()
            keywords = self._clean_string(attributes[4]).strip()

            # 权重可选
            weight = 1.0
            if len(attributes) > 5:
                try:
                    weight = float(self._clean_string(attributes[5]))
                except:
                    weight = 1.0

            if not source_entity or not target_entity:
                return None

            if source_entity == target_entity:
                return None

            relation_id = self._generate_relation_id(
                source_entity, target_entity, description
            )

            return {
                "relation_id": relation_id,
                "source_entity": source_entity,
                "target_entity": target_entity,
                "relation_type": "RELATED_TO",
                "description": description,
                "keywords": keywords,
                "weight": weight,
                "source_chunk_id": chunk_id,
                "document_id": document_id,
                "created_at": self._get_current_timestamp(),
            }

        except Exception as e:
            logger.error(f"处理关系抽取时出错: {str(e)}")
            return None

    def _clean_string(self, text: str) -> str:
        """清理字符串"""
        return text.strip().strip('"').strip("'")

    def _split_string_by_markers(self, text: str, markers: List[str]) -> List[str]:
        """按多个标记分割字符串"""
        parts = [text]
        for marker in markers:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(marker))
            parts = new_parts
        return [part.strip() for part in parts if part.strip()]

    def _generate_relation_id(
        self,
        source: str,
        target: str,
        relation_type: str = "RELATED_TO",
    ) -> str:
        """生成稳定的关系ID"""
        content = f"{source}|{target}|{relation_type}"
        return f"rel-{hashlib.md5(content.encode('utf-8')).hexdigest()[:16]}"

    def _get_current_timestamp(self) -> str:
        """获取当前时间的TIMESTAMP格式字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

