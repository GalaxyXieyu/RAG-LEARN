"""文件导入器"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FileImporter:
    """文件导入器
    
    支持多种文件格式的导入和解析（PDF, DOCX, TXT等）。
    """

    def __init__(self):
        """初始化文件导入器"""
        self.supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}

    def import_file(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Dict[str, Any]]:
        """导入文件并返回chunks
        
        Args:
            file_path: 文件路径
            chunk_size: chunk大小（字符数）
            chunk_overlap: chunk重叠大小
            
        Returns:
            List[Dict]: chunk列表，每个chunk包含：
                - text: 文本内容
                - chunk_id: chunk ID
                - source_file: 源文件路径
                - page: 页码（如果适用）
                - position: 位置信息
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            logger.warning(f"不支持的文件格式: {extension}，尝试作为文本文件处理")

        try:
            # 根据文件类型选择解析方法
            if extension == ".pdf":
                text_content = self._parse_pdf(file_path)
            elif extension in {".docx", ".doc"}:
                text_content = self._parse_docx(file_path)
            elif extension in {".txt", ".md"}:
                text_content = self._parse_text(file_path)
            else:
                # 默认按文本处理
                text_content = self._parse_text(file_path)

            # 分块处理
            chunks = self._chunk_text(
                text_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_file=str(file_path),
            )

            logger.info(f"成功导入文件 {file_path}，生成 {len(chunks)} 个chunks")
            return chunks

        except Exception as e:
            logger.error(f"导入文件失败 {file_path}: {str(e)}")
            raise

    def _parse_pdf(self, file_path: Path) -> str:
        """解析PDF文件"""
        try:
            # 尝试使用PyPDF2或pypdf
            try:
                import PyPDF2
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                try:
                    import pypdf
                    with open(file_path, "rb") as f:
                        pdf_reader = pypdf.PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except ImportError:
                    logger.warning("未安装PDF解析库，尝试使用其他方法")
                    raise ImportError("需要安装 PyPDF2 或 pypdf")
        except Exception as e:
            logger.error(f"PDF解析失败: {str(e)}")
            raise

    def _parse_docx(self, file_path: Path) -> str:
        """解析DOCX文件"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except ImportError:
            raise ImportError("需要安装 python-docx: pip install python-docx")
        except Exception as e:
            logger.error(f"DOCX解析失败: {str(e)}")
            raise

    def _parse_text(self, file_path: Path) -> str:
        """解析文本文件"""
        try:
            # 尝试多种编码
            encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"无法解码文件: {file_path}")
        except Exception as e:
            logger.error(f"文本文件解析失败: {str(e)}")
            raise

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        source_file: str = "",
    ) -> List[Dict[str, Any]]:
        """将文本分块
        
        Args:
            text: 文本内容
            chunk_size: chunk大小
            chunk_overlap: chunk重叠大小
            source_file: 源文件路径
            
        Returns:
            List[Dict]: chunk列表
        """
        chunks = []
        text_length = len(text)

        if text_length == 0:
            return chunks

        start = 0
        chunk_index = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]

            # 尝试在句子边界处切分
            if end < text_length:
                # 向后查找句号、问号、感叹号或换行符
                for i in range(end, min(end + 100, text_length)):
                    if text[i] in "。！？\n":
                        end = i + 1
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{source_file}_chunk_{chunk_index}"
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "chunk_order_index": chunk_index,
                    "source_file": source_file,
                    "tokens": len(chunk_text.split()),  # 简单的token计数
                    "page": 0,  # 默认值，可以在解析时设置
                    "position": f"{start}-{end}",
                })
                chunk_index += 1

            # 移动到下一个chunk的起始位置（考虑重叠）
            start = end - chunk_overlap
            if start < 0:
                start = end

        return chunks

