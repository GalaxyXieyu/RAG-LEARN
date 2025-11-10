"""
工程阶段分类统一封装

StageClassifier：纯 LLM 分类，移除规则判断逻辑。
暴露常量 STAGES 与 classify_text/classify_from_chunk 接口。
"""
from __future__ import annotations

from typing import List, Optional, Dict
from dataclasses import dataclass

from .llm import single_choice


STAGES: List[str] = [
    "土石方工程",
    "地基处理工程",
    "基坑支护工程",
    "主体工程",
    "装饰装修工程",
    "设备安装工程",
    "室外工程",
]


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: int = 20
    retries: int = 2


class StageClassifier:
    """工程阶段分类器（纯 LLM）"""
    
    def __init__(self, llm: Optional[LLMConfig] = None):
        """
        Args:
            llm: LLM 配置，如果为 None 则返回默认值 "主体工程"
        """
        self.llm = llm

    def classify_text(self, text: str, headers_norm: str, block_type: str) -> str:
        """
        使用 LLM 分类工程阶段
        
        Args:
            text: 文本内容
            headers_norm: 规范化的表头
            block_type: 块类型
        
        Returns:
            工程阶段（STAGES 中的一个）
        """
        # 如果没有配置 LLM，返回默认值
        if not self.llm:
            return "主体工程"
        
        try:
            # 准备上下文
            context = {
                "block_type": block_type,
                "headers_norm": headers_norm,
                "文本": (text or "")[:1200],  # 限制长度以降低成本
            }
            
            # 调用 LLM 进行单选分类
            pred = single_choice(
                labels=STAGES,
                context=context,
                base_url=self.llm.base_url,
                api_key=self.llm.api_key,
                model=self.llm.model,
                timeout=self.llm.timeout,
                retries=self.llm.retries,
            )
            
            # 验证结果
            if pred in STAGES:
                return pred
            
            # 如果返回的不在列表中，返回默认值
            return "主体工程"
            
        except Exception as e:
            # LLM 调用失败，返回默认值
            print(f"  ⚠️ LLM 分类失败: {e}，使用默认值")
            return "主体工程"

    def classify_from_chunk(self, chunk: Dict) -> str:
        """
        从 chunk 中提取信息并分类
        
        Args:
            chunk: chunk 字典
        
        Returns:
            工程阶段
        """
        text = chunk.get("text", "")
        headers_norm = "|".join(chunk.get("headers_norm", []) or [])
        block_type = chunk.get("type", "table")
        
        return self.classify_text(
            text=text,
            headers_norm=headers_norm,
            block_type=block_type
        )
