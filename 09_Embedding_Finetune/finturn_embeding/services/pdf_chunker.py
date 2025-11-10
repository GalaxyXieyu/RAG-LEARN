"""
PDF 分块处理服务
从 step2_ocr_chunking.py 和 LightweightChunker 抽取的核心逻辑
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from ..chunking.lightweight_chunker import LightweightChunker


class PDFChunker:
    """PDF 分块处理器"""
    
    def __init__(
        self,
        enable_ocr: bool = True,
        min_chunk_size: int = 100,
        extract_tables_only: bool = True,
        require_feature_col: bool = True,
        force_ocr: bool = False,
        enable_desc_from_non_table: bool = False,
    ):
        """
        Args:
            enable_ocr: 是否启用 OCR
            min_chunk_size: 最小 chunk 大小
            extract_tables_only: 仅提取表格
            require_feature_col: 要求表头包含"项目特征描述"列
            force_ocr: 强制使用 OCR
            enable_desc_from_non_table: 从非表格页抽取描述
        """
        self.chunker = LightweightChunker(
            enable_ocr=enable_ocr,
            min_chunk_size=min_chunk_size,
            extract_tables_only=extract_tables_only,
            require_feature_col=require_feature_col,
            force_ocr=force_ocr,
            enable_desc_from_non_table=enable_desc_from_non_table,
        )
    
    def process_pdf(
        self,
        pdf_path: Path,
        document_id: int,
    ) -> List[Dict[str, Any]]:
        """
        处理单个 PDF 文件
        
        Args:
            pdf_path: PDF 文件路径
            document_id: 文档ID
        
        Returns:
            chunks 列表
        """
        return self.chunker.process_pdf_file(str(pdf_path), document_id)
    
    def process_batch(
        self,
        pdf_files: List[Path],
        output_dir: Path,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        批量处理 PDF 文件
        
        Args:
            pdf_files: PDF 文件列表
            output_dir: 输出目录
            skip_existing: 是否跳过已处理的文件
        
        Returns:
            处理结果统计
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success = 0
        failed = 0
        skipped = 0
        
        for pdf_path in pdf_files:
            # 提取 document_id
            document_id = pdf_path.parent.name
            output_file = output_dir / f"{document_id}.json"
            
            # 跳过已处理
            if skip_existing and output_file.exists():
                skipped += 1
                continue
            
            try:
                # 处理 PDF
                chunks = self.process_pdf(pdf_path, document_id=0)
                
                if not chunks:
                    failed += 1
                    continue
                
                # 提取 markdown 表格
                markdown_tables = self._extract_markdown_tables(chunks)
                
                # 保存结果
                output_data = {
                    "document_id": document_id,
                    "source_file": pdf_path.name,
                    "chunks": chunks,
                    "markdown_tables": markdown_tables,
                    "total_chunks": len(chunks),
                    "total_tables": len(markdown_tables)
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                success += 1
                
            except Exception as e:
                failed += 1
                print(f"  处理失败 {document_id}: {e}")
        
        return {
            "total": len(pdf_files),
            "success": success,
            "failed": failed,
            "skipped": skipped
        }
    
    def _extract_markdown_tables(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """从 chunks 中提取 markdown 表格"""
        markdown_tables = []
        
        for chunk in chunks:
            if chunk.get("type") == "table":
                # 优先使用 table_markdown 字段
                md = chunk.get("table_markdown") or chunk.get("content", "")
                if md:
                    markdown_tables.append(md)
        
        return markdown_tables


def chunk_pdf(
    pdf_path: Path,
    document_id: int,
    enable_ocr: bool = True,
    min_chunk_size: int = 100,
    require_feature_col: bool = True,
) -> List[Dict[str, Any]]:
    """
    快捷函数：处理单个 PDF 文件
    
    Args:
        pdf_path: PDF 文件路径
        document_id: 文档ID
        enable_ocr: 是否启用 OCR
        min_chunk_size: 最小 chunk 大小
        require_feature_col: 要求表头包含"项目特征描述"列
    
    Returns:
        chunks 列表
    """
    chunker = PDFChunker(
        enable_ocr=enable_ocr,
        min_chunk_size=min_chunk_size,
        require_feature_col=require_feature_col,
    )
    return chunker.process_pdf(pdf_path, document_id)

