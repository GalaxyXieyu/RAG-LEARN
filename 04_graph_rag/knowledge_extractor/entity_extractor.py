"""实体抽取器"""

import re
import hashlib
import logging
from typing import Dict, List, Optional
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


class EntityExtractor:
    """实体抽取器
    
    从文本中抽取实体信息。支持通过PromptManager灵活配置业务提示词。
    """

    def __init__(
        self,
        model_type: Optional[LlmClientType] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """初始化实体抽取器
        
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
        """从文本中抽取实体
        
        Args:
            content: 文本内容
            chunk_id: chunk ID
            document_id: 文档ID
            
        Returns:
            Dict: 实体字典，key为entity_name，value为实体数据列表
        """
        try:
            # 准备提示词
            examples = self.prompt_manager.get_examples()
            examples_str = "\n".join([
                ex.format(
                    tuple_delimiter=self.prompt_manager.tuple_delimiter,
                    record_delimiter=self.prompt_manager.record_delimiter,
                    completion_delimiter=self.prompt_manager.completion_delimiter,
                ) if isinstance(ex, str) else str(ex)
                for ex in examples
            ])
            
            context_base = {
                "tuple_delimiter": self.prompt_manager.tuple_delimiter,
                "record_delimiter": self.prompt_manager.record_delimiter,
                "completion_delimiter": self.prompt_manager.completion_delimiter,
                "entity_types": ",".join(self.prompt_manager.entity_types),
                "examples": examples_str,
                "language": self.prompt_manager.language,
                "input_text": content,
            }

            prompt = self.prompt_manager.get_prompt("entity_extraction", **context_base)

            # 调用LLM
            messages = [HumanMessage(content=prompt)]
            response = await chat_async(
                model_type=self.model_type,
                messages=messages,
                temperature=0.1,
                max_tokens=4000,
                stream=False,
            )

            response_content = response.content.strip() if response.content else ""
            if not response_content:
                logger.error(f"LLM返回空响应，跳过chunk {chunk_id}")
                return {}

            # 解析抽取结果
            entities = await self._parse_extraction_result(
                response_content, chunk_id, document_id
            )

            return entities

        except Exception as e:
            logger.error(f"实体抽取失败 (chunk {chunk_id}): {str(e)}")
            return {}

    async def _parse_extraction_result(
        self,
        response: str,
        chunk_id: str,
        document_id: int,
    ) -> Dict:
        """解析LLM的抽取结果"""
        entities = defaultdict(list)

        if not response:
            return dict(entities)

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

            if len(attributes) < 4:
                continue

            # 处理实体
            if '"entity"' in attributes[0]:
                entity_data = await self._handle_entity_extraction(
                    attributes, chunk_id, document_id
                )
                if entity_data:
                    entities[entity_data["entity_name"]].append(entity_data)

        return dict(entities)

    async def _handle_entity_extraction(
        self,
        attributes: List[str],
        chunk_id: str,
        document_id: int,
    ) -> Optional[Dict]:
        """处理单个实体抽取结果"""
        try:
            if len(attributes) < 4:
                return None

            entity_name = self._clean_string(attributes[1]).strip()
            entity_type = self._clean_string(attributes[2]).strip()
            description = self._clean_string(attributes[3]).strip()

            if not entity_name or not entity_type:
                return None

            entity_id = self._generate_entity_id(entity_name)

            return {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "entity_type": entity_type,
                "description": description,
                "source_chunk_id": chunk_id,
                "document_id": document_id,
                "created_at": self._get_current_timestamp(),
            }

        except Exception as e:
            logger.error(f"处理实体抽取时出错: {str(e)}")
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

    def _generate_entity_id(self, entity_name: str) -> str:
        """生成稳定的实体ID"""
        return f"ent-{hashlib.md5(entity_name.encode('utf-8')).hexdigest()[:16]}"

    def _get_current_timestamp(self) -> str:
        """获取当前时间的TIMESTAMP格式字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

