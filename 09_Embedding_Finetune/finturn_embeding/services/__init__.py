"""
服务层：提供原子能力
- report_fetcher: 报告文件获取
- pdf_chunker: PDF 分块处理
- table_merger: 跨页表格合并
"""

from .report_fetcher import fetch_reports, ReportFetcher
from .pdf_chunker import chunk_pdf, PDFChunker
from .table_merger import merge_table_groups, TableMerger

__all__ = [
    'fetch_reports',
    'ReportFetcher',
    'chunk_pdf',
    'PDFChunker',
    'merge_table_groups',
    'TableMerger',
]

