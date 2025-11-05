"""关键词提取器"""

import json
import re
import logging
from typing import List, Tuple, Optional

from langchain_core.messages import HumanMessage
from app.llm.base_chat import chat_async
from app.core.config.config import LlmClientType

try:
    from ..knowledge_extractor.prompt_manager import PromptManager
except ImportError:
    from knowledge_extractor.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """关键词提取器
    
    从查询中提取高层关键词和低层关键词。支持通过PromptManager灵活配置提示词。
    """

    def __init__(
        self,
        model_type=None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """初始化关键词提取器
        
        Args:
            model_type: LLM客户端类型
            prompt_manager: 提示词管理器实例（可选）
        """
        if model_type is None:
            model_type = LlmClientType.DeepSeekV3
        self.model_type = model_type
        
        # 使用提供的prompt_manager或创建默认的
        self.prompt_manager = prompt_manager or PromptManager()

    async def extract_keywords(self, query: str, history: str = "") -> Tuple[List[str], List[str]]:
        """从查询中提取高层关键词和低层关键词
        
        Args:
            query: 用户查询文本
            history: 历史对话（可选）
            
        Returns:
            Tuple[List[str], List[str]]: (high_level_keywords, low_level_keywords)
            - high_level_keywords: 用于关系检索的抽象概念
            - low_level_keywords: 用于实体检索的具体实体
        """
        try:
            # 使用提示词管理器获取关键词提取提示词
            examples = self.prompt_manager.get_config("keywords_extraction_examples", [])
            examples_str = "\n".join(examples) if isinstance(examples, list) else str(examples)

            keyword_prompt = self.prompt_manager.get_prompt(
                "keywords_extraction",
                examples=examples_str,
                history=history,
                query=query
            )

            # 调用LLM提取关键词
            messages = [HumanMessage(content=keyword_prompt)]
            response = await chat_async(
                model_type=self.model_type,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
                stream=False,
            )

            if not response.content:
                logger.warning("关键词提取返回空响应")
                return [], []

            # 解析JSON响应
            try:
                response_text = response.content.strip()
                logger.debug(f"原始响应内容: {response_text[:200]}...")

                # 提取JSON部分，处理R1模型的<think>标签
                start_pos = response_text.find("{")
                end_pos = response_text.rfind("}") + 1

                if start_pos == -1 or end_pos == -1:
                    logger.error("无法在响应中找到有效的JSON部分")
                    return [], []

                json_part = response_text[start_pos:end_pos]

                # 修复常见的JSON格式问题
                # 1. 移除注释
                json_part = re.sub(r"//.*", "", json_part)  # 移除单行注释
                json_part = re.sub(
                    r"/\*.*?\*/", "", json_part, flags=re.DOTALL
                )  # 移除多行注释

                # 2. 修复字符串中的转义问题
                json_part = json_part.replace('\\"', '"').replace("\\'", "'")

                # 3. 移除结尾多余的逗号
                json_part = re.sub(r",\s*}", "}", json_part)
                json_part = re.sub(r",\s*]", "]", json_part)

                logger.debug(f"修复后的JSON: {json_part[:200]}...")

                keywords_data = json.loads(json_part)

                high_level_keywords = keywords_data.get("high_level_keywords", [])
                low_level_keywords = keywords_data.get("low_level_keywords", [])

                # 确保返回的是字符串列表
                if not isinstance(high_level_keywords, list):
                    high_level_keywords = []
                if not isinstance(low_level_keywords, list):
                    low_level_keywords = []

                # 过滤空值
                high_level_keywords = [
                    kw for kw in high_level_keywords if kw and kw.strip()
                ]
                low_level_keywords = [
                    kw for kw in low_level_keywords if kw and kw.strip()
                ]

                logger.info(
                    f"关键词提取结果 - 高层: {high_level_keywords}, 低层: {low_level_keywords}"
                )
                return high_level_keywords, low_level_keywords

            except json.JSONDecodeError as e:
                logger.error(
                    f"关键词提取JSON解析失败: {str(e)}, 响应内容: {response.content[:200] if response.content else ''}"
                )
                return [], []

        except Exception as e:
            logger.error(f"关键词提取失败: {str(e)}")
            return [], []

