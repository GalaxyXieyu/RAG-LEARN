"""文档分块器"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Chunker:
    """文档分块器
    
    支持多种分块策略。
    """

    def __init__(self, strategy: str = "fixed"):
        """初始化分块器
        
        Args:
            strategy: 分块策略，支持 "fixed"（固定大小）、"sentence"（按句子）、"paragraph"（按段落）
        """
        self.strategy = strategy

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """分块文本
        
        Args:
            text: 要分块的文本
            chunk_size: chunk大小（字符数或token数）
            chunk_overlap: chunk重叠大小
            **kwargs: 其他参数
            
        Returns:
            List[Dict]: chunk列表
        """
        if self.strategy == "fixed":
            return self._chunk_fixed_size(text, chunk_size, chunk_overlap, **kwargs)
        elif self.strategy == "sentence":
            return self._chunk_by_sentence(text, chunk_size, **kwargs)
        elif self.strategy == "paragraph":
            return self._chunk_by_paragraph(text, chunk_size, **kwargs)
        else:
            logger.warning(f"未知的分块策略: {self.strategy}，使用固定大小策略")
            return self._chunk_fixed_size(text, chunk_size, chunk_overlap, **kwargs)

    def _chunk_fixed_size(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """固定大小分块"""
        chunks = []
        text_length = len(text)

        if text_length == 0:
            return chunks

        start = 0
        chunk_index = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": kwargs.get("chunk_id_prefix", "chunk") + f"_{chunk_index}",
                    "chunk_order_index": chunk_index,
                    "tokens": len(chunk_text.split()),
                    "position": f"{start}-{end}",
                })
                chunk_index += 1

            start = end - chunk_overlap
            if start < 0:
                start = end

        return chunks

    def _chunk_by_sentence(
        self,
        text: str,
        max_chunk_size: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """按句子分块"""
        import re

        # 按句子分割（支持中英文）
        sentences = re.split(r'[。！？\n]+|\.\s+|!\s+|\?\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)
            if current_size + sentence_size > max_chunk_size and current_chunk:
                # 保存当前chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": kwargs.get("chunk_id_prefix", "chunk") + f"_{chunk_index}",
                    "chunk_order_index": chunk_index,
                    "tokens": len(chunk_text.split()),
                })
                chunk_index += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        # 添加最后一个chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "chunk_id": kwargs.get("chunk_id_prefix", "chunk") + f"_{chunk_index}",
                "chunk_order_index": chunk_index,
                "tokens": len(chunk_text.split()),
            })

        return chunks

    def _chunk_by_paragraph(
        self,
        text: str,
        max_chunk_size: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """按段落分块"""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_size = len(paragraph)
            if current_size + paragraph_size > max_chunk_size and current_chunk:
                # 保存当前chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": kwargs.get("chunk_id_prefix", "chunk") + f"_{chunk_index}",
                    "chunk_order_index": chunk_index,
                    "tokens": len(chunk_text.split()),
                })
                chunk_index += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(paragraph)
            current_size += paragraph_size

        # 添加最后一个chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "chunk_id": kwargs.get("chunk_id_prefix", "chunk") + f"_{chunk_index}",
                "chunk_order_index": chunk_index,
                "tokens": len(chunk_text.split()),
            })

        return chunks

