"""知识抽取器 - 统一的实体和关系抽取接口"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from langchain_core.messages import HumanMessage
from app.llm.base_chat import chat_async
from app.core.config.config import LlmClientType

try:
    from .entity_extractor import EntityExtractor
    from .relation_extractor import RelationExtractor
    from .prompt_manager import PromptManager
except ImportError:
    from entity_extractor import EntityExtractor
    from relation_extractor import RelationExtractor
    from prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """知识抽取器
    
    统一管理实体抽取和关系抽取。在实际实现中，实体和关系在一个LLM调用中一起抽取。
    支持通过PromptManager灵活配置业务提示词。
    """

    def __init__(
        self,
        model_type=None,
        prompt_manager: Optional[PromptManager] = None,
        prompt_file_path: Optional[str] = None,
        entity_extractor: EntityExtractor = None,
        relation_extractor: RelationExtractor = None,
    ):
        """初始化知识抽取器
        
        Args:
            model_type: LLM客户端类型
            prompt_manager: 提示词管理器实例（可选）
            prompt_file_path: 提示词文件路径（可选，如果提供则创建PromptManager）
            entity_extractor: 实体抽取器实例（用于解析）
            relation_extractor: 关系抽取器实例（用于解析）
        """
        if model_type is None:
            model_type = LlmClientType.DeepSeekV3
        self.model_type = model_type
        
        # 初始化提示词管理器
        if prompt_manager:
            self.prompt_manager = prompt_manager
        elif prompt_file_path:
            self.prompt_manager = PromptManager(prompt_file_path=prompt_file_path)
        else:
            # 默认加载标准提示词
            self.prompt_manager = PromptManager()
        
        # 用于解析的抽取器（共享相同的prompt_manager）
        self.entity_extractor = entity_extractor or EntityExtractor(
            model_type=model_type,
            prompt_manager=self.prompt_manager
        )
        self.relation_extractor = relation_extractor or RelationExtractor(
            model_type=model_type,
            prompt_manager=self.prompt_manager
        )

    async def extract(
        self,
        content: str,
        chunk_id: str,
        document_id: int,
    ) -> Tuple[Dict, Dict]:
        """从文本中抽取实体和关系
        
        Args:
            content: 文本内容
            chunk_id: chunk ID
            document_id: 文档ID
            
        Returns:
            Tuple[Dict, Dict]: (entities_dict, relations_dict)
        """
        try:
            # 使用统一的LLM调用来抽取实体和关系
            entities, relations = await self._extract_from_chunk(
                content, chunk_id, document_id
            )
            return entities, relations
        except Exception as e:
            logger.error(f"知识抽取失败 (chunk {chunk_id}): {str(e)}")
            return {}, {}

    async def _extract_from_chunk(
        self,
        content: str,
        chunk_id: str,
        document_id: int,
    ) -> Tuple[Dict, Dict]:
        """从单个chunk中抽取实体和关系
        
        使用一个LLM调用来同时抽取实体和关系。
        """
        try:
            # 准备提示词模板的参数
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

            # 使用提示词管理器获取格式化的提示词
            prompt = self.prompt_manager.get_prompt("entity_extraction", **context_base)

            # 直接调用LLM
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
                return {}, {}

            # 解析抽取结果
            entities, relations = await self._parse_extraction_result(
                response_content, chunk_id, document_id
            )

            return entities, relations

        except Exception as e:
            logger.error(f"抽取chunk {chunk_id} 时出错: {str(e)}")
            return {}, {}

    async def _parse_extraction_result(
        self,
        response: str,
        chunk_id: str,
        document_id: int,
    ) -> Tuple[Dict, Dict]:
        """解析LLM的抽取结果"""
        entities = defaultdict(list)
        relations = defaultdict(list)

        if not response:
            logger.warning(f"LLM返回空响应，跳过chunk {chunk_id}")
            return dict(entities), dict(relations)

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
                entity_data = await self.entity_extractor._handle_entity_extraction(
                    attributes, chunk_id, document_id
                )
                if entity_data:
                    entities[entity_data["entity_name"]].append(entity_data)

            # 处理关系
            elif '"relationship"' in attributes[0]:
                relation_data = await self.relation_extractor._handle_relation_extraction(
                    attributes, chunk_id, document_id
                )
                if relation_data:
                    relation_key = (
                        relation_data["source_entity"],
                        relation_data["target_entity"],
                    )
                    relations[relation_key].append(relation_data)

        return dict(entities), dict(relations)

    def _split_string_by_markers(self, text: str, markers: List[str]) -> List[str]:
        """按多个标记分割字符串"""
        parts = [text]
        for marker in markers:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(marker))
            parts = new_parts
        return [part.strip() for part in parts if part.strip()]

