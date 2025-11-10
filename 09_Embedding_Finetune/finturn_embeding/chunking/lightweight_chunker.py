"""
轻量级文档分块器 - 基于 PyMuPDF + RapidOCR
专门用于 files_extraction 功能，作为 MinerU 的轻量级替代方案

新增能力：
1) 增加“描述块（desc）”抽取：围绕表格的上/下方说明性文本，用于因子描述检索。
2) 表格表头规范化与分组：输出 headers_norm、table_group_id，支撑后续“并表”。
3) 因子命中统计（可选）：加载因子 JSON 做轻量匹配，写入 factor_refs（仅 CLI 输出使用）。
4) CLI 预览与 CSV 导出：支持 --preview 打印与 --save-csv 落盘，便于人工分析与标注准备。

核心特性：
1. 纯 CPU 运行，速度快（5-10 倍于 MinerU）
2. 自动检测并提取表格
3. 筛选包含「项目特征描述」列的表格
4. 输出格式与 MinerU 保持一致
5. 支持扫描 PDF 检测与 OCR 识别（可选）

使用场景：
- 低算力环境
- 快速文档处理
- 支持扫描版 PDF（启用 OCR）
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import pymupdf  # PyMuPDF
import hashlib
import csv
import json
import argparse

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False


class LightweightChunker:
    """PyMuPDF轻量级分块器 - 提取表格并生成chunks（支持OCR）"""
    
    def __init__(self, 
                 filter_keywords: Optional[List[str]] = None,
                 min_chunk_size: int = 50,
                 extract_tables_only: bool = True,
                 max_pages: Optional[int] = None,
                 enable_ocr: bool = False,
                 require_feature_col: bool = True,
                 force_ocr: bool = False,
                 enable_desc_from_non_table: bool = False):
        """
        初始化轻量级分块器
        
        Args:
            filter_keywords: 表格内容关键词过滤（None=不使用关键词过滤）
            min_chunk_size: 最小chunk大小（字符数）
            extract_tables_only: 仅提取表格（True）或提取所有内容（False）
            max_pages: 最大处理页数（None=处理全部）
            enable_ocr: 是否启用OCR（遇到扫描版PDF时自动使用）
        """
        self.filter_keywords = filter_keywords
        self.min_chunk_size = min_chunk_size
        self.extract_tables_only = extract_tables_only
        self.max_pages = max_pages
        self.enable_ocr = enable_ocr
        self.require_feature_col = require_feature_col
        self.force_ocr = force_ocr
        self.enable_desc_from_non_table = enable_desc_from_non_table
        
        # 初始化 OCR 引擎
        self.ocr_engine = None
        if self.enable_ocr and RAPIDOCR_AVAILABLE:
            try:
                self.ocr_engine = RapidOCR()
            except Exception as e:
                self.ocr_engine = None
    
    def process_pdf_file(self, pdf_path: str, document_id: int) -> List[Dict[str, Any]]:
        """
        处理PDF文件，提取表格并生成chunks（自动检测扫描版并使用OCR）
        
        Args:
            pdf_path: PDF文件路径
            document_id: 文档ID
            
        Returns:
            包含筛选后表格的chunk列表
            
        Raises:
            ValueError: 扫描PDF且OCR未启用或不可用
            FileNotFoundError: PDF文件不存在
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        try:
            doc = pymupdf.open(str(pdf_file))
            
            # 检测是否为扫描PDF（可被 force_ocr 覆盖）
            is_scanned = self._is_scanned_pdf(doc)
            
            if is_scanned or self.force_ocr:
                # 扫描版PDF处理
                if not self.enable_ocr or not self.ocr_engine:
                    doc.close()
                    if not self.enable_ocr:
                        raise ValueError(
                            f"文档{document_id}是扫描版PDF，但OCR未启用。"
                            "请在配置中启用 enable_ocr=True"
                        )
                    else:
                        raise ValueError(
                            f"文档{document_id}是扫描版PDF，但OCR引擎不可用。"
                            "请安装: pip install rapidocr-onnxruntime"
                        )
                
                # 使用OCR处理
                chunks = self._process_scanned_pdf(doc, document_id, pdf_path)
            else:
                # 正常PDF处理
                chunks = self._process_document(doc, document_id)
            
            doc.close()
            
            return chunks
            
        except Exception as e:
            print(f"处理PDF时出错: {pdf_path}, 错误: {e}")
            raise
    
    def _is_scanned_pdf(self, doc: pymupdf.Document) -> bool:
        """
        检测是否为扫描版PDF（无文本层）
        
        策略：如果前10页中少于3页有文本（>50字符），判定为扫描版
        """
        text_pages = 0
        check_pages = min(10, doc.page_count)
        
        for i in range(check_pages):
            try:
                text = doc[i].get_text('text').strip()
                if len(text) > 50:
                    text_pages += 1
            except Exception as e:
                continue
        
        return text_pages < 3
    
    def _process_scanned_pdf(self, doc: pymupdf.Document, document_id: int, pdf_path: str) -> List[Dict]:
        """
        处理扫描版PDF（使用OCR）
        
        Args:
            doc: PyMuPDF文档对象
            document_id: 文档ID
            pdf_path: PDF文件路径
            
        Returns:
            包含OCR识别结果的chunk列表
        """
        chunks = []
        chunk_index = 0
        
        # 确定处理页数
        max_page = self.max_pages if self.max_pages else doc.page_count
        process_pages = min(max_page, doc.page_count)
        
        for page_idx in range(process_pages):
            try:
                page = doc[page_idx]
                
                # OCR识别页面
                page_text, ocr_result = self._ocr_page(page, page_idx)
                
                if not page_text or len(page_text.strip()) < 50:
                    continue
                
                # 尝试从OCR结果中提取表格
                page_chunks = self._extract_tables_from_ocr(
                    page_text, ocr_result, page_idx, chunk_index, document_id
                )
                
                chunks.extend(page_chunks)
                chunk_index += len(page_chunks)
               
            except Exception as e:
                continue
        
        return chunks
    
    def _ocr_page(self, page, page_idx: int) -> Tuple[str, Any]:
        """
        对单个页面进行OCR识别
        
        Args:
            page: PyMuPDF页面对象
            page_idx: 页面索引
            
        Returns:
            (识别的完整文本, OCR原始结果)
        """
        try:
            # 将页面渲染为图片（提高分辨率以改善OCR效果）
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2倍缩放
            img_data = pix.tobytes("png")
            
            # 使用RapidOCR识别
            result, elapse = self.ocr_engine(img_data)
            
            if not result:
                return "", None
            
            # 提取文本（result格式: [[bbox, text, confidence], ...]）
            texts = [item[1] for item in result if item and len(item) > 1]
            full_text = "\n".join(texts)
            
            return full_text, result
            
        except Exception as e:
            print(f"OCR识别第{page_idx+1}页失败: {e}")
            return "", None
    
    def _extract_tables_from_ocr(self, page_text: str, ocr_result: Any, 
                                  page_idx: int, start_chunk_idx: int,
                                  document_id: int) -> List[Dict]:
        """
        从OCR结果中提取表格信息
        
        注意：扫描PDF的表格提取比较复杂，这里使用简化策略：
        1. 检查是否包含"项目特征描述"关键词
        2. 如果包含，将整页文本作为一个chunk
        3. 尝试识别工程类型
        
        Args:
            page_text: OCR识别的完整文本
            ocr_result: OCR原始结果
            page_idx: 页面索引
            start_chunk_idx: 起始chunk索引
            document_id: 文档ID
            
        Returns:
            chunk列表
        """
        chunks = []
        
        # 检查是否包含"项目特征描述"（可选核心筛选条件）
        if self.require_feature_col and ('项目特征描述' not in page_text and '特征描述' not in page_text):
            return chunks
        
        # 可选：关键词过滤
        if self.filter_keywords and not self._contains_keywords(page_text):
            return chunks
        
        # 检查内容长度
        if len(page_text) < self.min_chunk_size:
            return chunks
        
        # 提取工程类型
        construction_stage = self._extract_construction_stage(page_text, None)
        
        # 创建chunk（将整页作为一个表格chunk）
        chunk = self._create_ocr_chunk(
            page_text=page_text,
            page_idx=page_idx,
            chunk_index=start_chunk_idx,
            construction_stage=construction_stage,
            document_id=document_id
        )
        
        chunks.append(chunk)
        
        return chunks
    
    def _create_ocr_chunk(self, page_text: str, page_idx: int,
                         chunk_index: int, construction_stage: str,
                         document_id: int) -> Dict[str, Any]:
        """
        从OCR结果创建chunk
        
        Args:
            page_text: OCR识别的文本
            page_idx: 页码
            chunk_index: chunk索引
            construction_stage: 工程类型
            document_id: 文档ID
        """
        # 生成带工程类型的完整文本
        text_parts = []
        if construction_stage:
            text_parts.append(f"【{construction_stage}】")
        text_parts.append(f"第{page_idx+1}页（OCR识别）")
        text_parts.append(page_text)
        full_text = "\n".join(text_parts)
        
        # 生成Markdown格式内容
        content_parts = []
        if construction_stage:
            content_parts.append(f"## 【{construction_stage} - 第{page_idx+1}页】\n")
        else:
            content_parts.append(f"## 第{page_idx+1}页（OCR识别）\n")
        content_parts.append(page_text)
        content_markdown = "\n".join(content_parts)
        
        # 构建chunk
        chunk = {
            # 核心字段
            "construction_stage": construction_stage,
            "content": content_markdown,
            "text": full_text,
            "type": "table",  # 标记为table类型（虽然是OCR识别的）
            "page_idx": page_idx,
            "bbox": [],
            "chunk_index": chunk_index,
            
            # 表格特有字段
            "table_caption": "",
            "table_footnote": "",
            "table_body_html": "",
            "img_path": "",
            
            # 元数据
            "metadata": {
                "source": "rapidocr",
                "is_table": True,
                "has_caption": False,
                "has_footnote": False,
                "has_feature_column": True,
                "table_rows": 0,
                "table_cols": 0,
                "text_length": len(full_text),
                "construction_stage": construction_stage,
                "document_id": str(document_id),
                "is_ocr": True
            }
        }
        
        return chunk
    
    def _process_document(self, doc: pymupdf.Document, document_id: int) -> List[Dict]:
        """处理整个文档，提取所有符合条件的表格"""
        chunks: List[Dict[str, Any]] = []
        chunk_index = 0
        
        # 确定处理页数
        max_page = self.max_pages if self.max_pages else doc.page_count
        process_pages = min(max_page, doc.page_count)
        
        for page_idx in range(process_pages):
            try:
                page = doc[page_idx]
                
                # 提取页面中的表格
                page_chunks = self._extract_page_tables(page, page_idx, chunk_index, document_id)
                chunks.extend(page_chunks)
                chunk_index += len(page_chunks)

                # 若当前页未产生任何描述块，且允许从非表格页抽取描述，则尝试抽取
                if self.enable_desc_from_non_table:
                    has_desc = any(c.get("type") == "desc" and c.get("page_idx") == page_idx for c in chunks[-len(page_chunks):])
                    if not has_desc and (not page_chunks or True):
                        extra_desc = self._extract_desc_from_page(page, page_idx, chunk_index, document_id)
                        if extra_desc:
                            chunks.append(extra_desc)
                            chunk_index += 1
                
            except Exception as e:
                continue
        
        return chunks
    
    def _extract_page_tables(self, page, page_idx: int, 
                            start_chunk_idx: int, document_id: int) -> List[Dict]:
        """
        提取页面中的表格并生成chunks
        
        筛选条件：
        1. 表格必须包含"项目特征描述"列（核心条件）
        2. 可选：包含指定关键词（如果设置了filter_keywords）
        """
        chunks = []
        
        # 1. 检测表格
        try:
            tables = page.find_tables()
        except Exception as e:
            return chunks
        
        if not tables.tables:
            return chunks
        
        # 2. 提取页面文本（用于工程类型提取）
        page_text = page.get_text('text')
        page_blocks = []
        try:
            page_blocks = page.get_text("blocks")
        except Exception:
            page_blocks = []
        
        # 3. 遍历表格
        for table_idx, table in enumerate(tables.tables):
            try:
                # 3.1 提取表格数据
                data = table.extract()
                if not data or len(data) < 2:  # 至少需要表头+1行数据
                    continue
                
                # 3.2 检查表头（核心筛选条件）
                header = data[0]
                header_text = ' '.join([str(c) for c in header if c])
                
                if self.require_feature_col and not self._has_feature_description_column(header_text):
                    continue
                
                # 3.3 过滤空表/占位表（仅表头、空行、重复表头、仅小计/合计等）
                raw_header = [str(c).strip() for c in (data[0] if data else [])]
                headers_norm = [self._normalize_header(h) for h in raw_header if h]
                if self._is_placeholder_table(data, headers_norm):
                    continue

                # 3.4 生成并清理 Markdown（移除全空行）
                table_markdown = self._table_markdown_clean(table)
                if len(table_markdown) < self.min_chunk_size:
                    continue
                
                # 3.5 可选：关键词过滤
                if self.filter_keywords and not self._contains_keywords(table_markdown):
                    continue
                
                # 3.6 提取工程类型
                construction_stage = self._extract_construction_stage(page_text, table)
                
                # 3.7 创建chunk
                chunk = self._create_chunk_from_table(
                    table=table,
                    data=data,
                    page_idx=page_idx,
                    chunk_index=start_chunk_idx + len(chunks),
                    construction_stage=construction_stage,
                    document_id=document_id
                )
                
                chunks.append(chunk)

                # 3.8 创建对应的“描述块”（选取表格附近的说明文本）
                try:
                    desc_chunk = self._create_desc_chunk(
                        page=page,
                        page_blocks=page_blocks,
                        table_bbox=list(table.bbox) if hasattr(table, 'bbox') else None,
                        page_idx=page_idx,
                        chunk_index=start_chunk_idx + len(chunks),
                        construction_stage=construction_stage,
                        document_id=document_id,
                        ref_table_chunk=chunk
                    )
                    if desc_chunk:
                        chunks.append(desc_chunk)
                except Exception:
                    pass
                
            except Exception as e:
                continue
        
        return chunks
    
    def _has_feature_description_column(self, header_text: str) -> bool:
        """检查表头是否包含"项目特征描述"列（核心筛选条件）"""
        return '项目特征描述' in header_text or '特征描述' in header_text
    
    def _contains_keywords(self, text: str) -> bool:
        """检查文本是否包含任一筛选关键词"""
        if not self.filter_keywords:
            return True
        
        for keyword in self.filter_keywords:
            if keyword in text:
                return True
        return False
    
    def _extract_construction_stage(self, page_text: str, table) -> str:
        """
        提取工程类型（construction_stage）
        
        优先级：
        1. 表格上方文本中的"工程名称:xxx"
        2. 页面文本中的"工程名称:xxx"（前500字符）
        3. 默认返回空字符串
        """
        # 方法1: 检查表格bbox上方的文本（前1000字符）
        match = re.search(r'工程名称[:\s：]+([^\s，。；标]+(?:工程)?)', page_text[:1000])
        if match:
            stage = match.group(1).strip()
            stage = re.sub(r'[，。；：:\s]+$', '', stage)
            return stage if stage else ""
        
        # 方法2: 检查整个页面（前500字符）
        match = re.search(r'工程名称[:\s：]+([^\s，。；标]+(?:工程)?)', page_text[:500])
        if match:
            stage = match.group(1).strip()
            stage = re.sub(r'[，。；：:\s]+$', '', stage)
            return stage if stage else ""
        
        return ""
    
    def _create_chunk_from_table(self, table, data: List, page_idx: int, 
                                 chunk_index: int, construction_stage: str,
                                 document_id: int) -> Dict[str, Any]:
        """
        从表格创建chunk（格式与MinerU保持一致）
        
        Args:
            table: PyMuPDF Table对象
            data: 表格数据（二维数组）
            page_idx: 页码
            chunk_index: chunk索引
            construction_stage: 工程类型
            document_id: 文档ID
        """
        # 1. 生成Markdown内容（并清理空行）
        table_markdown = self._table_markdown_clean(table)
        
        # 2. 生成纯文本（用于向量化）
        # 包含工程类型信息，提升检索准确度
        text_parts = []
        if construction_stage:
            text_parts.append(f"【{construction_stage}】")
        text_parts.append(f"第{page_idx+1}页表格")
        text_parts.append(table_markdown)
        full_text = "\n".join(text_parts)
        
        # 3. 生成content（Markdown格式，用于展示）
        content_parts = []
        if construction_stage:
            content_parts.append(f"## 【{construction_stage} - 第{page_idx+1}页】\n")
        else:
            content_parts.append(f"## 第{page_idx+1}页表格\n")
        content_parts.append(table_markdown)
        content_markdown = "\n".join(content_parts)
        
        # 4. 规范化表头（用于后续并表）
        raw_header = [str(c).strip() for c in (data[0] if data else [])]
        headers_norm = [self._normalize_header(h) for h in raw_header if h]
        # 生成分组 id（headers_norm + stage）
        group_seed = "|".join(headers_norm) + f"|{construction_stage or ''}"
        table_group_id = hashlib.md5(group_seed.encode("utf-8")).hexdigest()[:12]

        # 5. 构建chunk（与MinerU格式保持一致）
        chunk = {
            # 核心字段
            "construction_stage": construction_stage,
            "content": content_markdown,
            "text": full_text,
            "type": "table",
            "page_idx": page_idx,
            "bbox": list(table.bbox) if hasattr(table, 'bbox') else [],
            "chunk_index": chunk_index,
            
            # 表格特有字段
            "table_caption": "",  # PyMuPDF不直接提供，留空
            "table_footnote": "",
            "table_body_html": "",  # 简化实现，留空
            "img_path": "",
            "headers_norm": headers_norm,
            "table_group_id": table_group_id,
            "table_markdown": table_markdown,
            
            # 元数据
            "metadata": {
                "source": "pymupdf",
                "is_table": True,
                "has_caption": False,
                "has_footnote": False,
                "has_feature_column": True,  # 已筛选过
                "table_rows": table.row_count,
                "table_cols": table.col_count,
                "text_length": len(full_text),
                "construction_stage": construction_stage,
                "document_id": str(document_id)
            }
        }
        
        return chunk

    # ========= 新增：描述块与辅助 =========
    def _normalize_header(self, h: str) -> str:
        """规范化表头名称：去空白、全角转半角、统一常见同义词。"""
        s = (h or "").strip()
        s = re.sub(r"[\u3000\s]+", " ", s)
        s = s.replace("：", ":").replace("（", "(").replace("）", ")")
        s = s.lower()
        synonyms = {
            "建筑面积": "area",
            "地上建筑面积": "area",
            "面积": "area",
            "层数": "floors",
            "檐高": "floors",
            "处理深度": "depth",
            "挖槽深度": "depth",
            "桩深": "depth",
            "桩径": "diameter",
            "工程量": "quantity",
            "计量单位": "unit",
            "单位": "unit",
            "项目名称": "item_name",
            "项目特征描述": "feature_desc",
            "特征描述": "feature_desc",
        }
        return synonyms.get(s, s)

    def _is_placeholder_table(self, data: List[List[Any]], headers_norm: List[str]) -> bool:
        """判断是否为空表/占位表：
        规则（满足任一判为占位）：
        1) 含有 feature_desc 列，但该列在数据区无任何有效内容；
        2) 数据区非空单元格比例 < 5%；
        3) 数据区仅包含“合计/本页小计”等汇总行；
        4) 数据区存在大量重复表头行（误检将表头识别到正文）。
        """
        try:
            if not data or len(data) < 2:
                return True
            header = [str(c or '').strip() for c in data[0]]
            body = data[1:]

            # 映射关键列索引
            idx_feature = [i for i, h in enumerate(headers_norm) if h == 'feature_desc']
            idx_unit = [i for i, h in enumerate(headers_norm) if h == 'unit']
            idx_quantity = [i for i, h in enumerate(headers_norm) if h == 'quantity']

            def clean_cell(x: Any) -> str:
                s = str(x or '').strip()
                s = re.sub(r"[\u00A0\s]+", " ", s)  # 空白规范化
                s = s.replace("\n", " ")
                s = re.sub(r"\|+", "|", s)  # 竖线规整
                return s

            def is_effective_text(s: str) -> bool:
                if not s:
                    return False
                t = s.strip().strip('|').strip('-').strip()
                if not t:
                    return False
                # 常见占位符/线条
                if re.fullmatch(r"[-—·\.\s]+", t):
                    return False
                # 重复表头词 & 汇总词
                bad_tokens = {'序号','项目编码','项目名称','项目特征描述','特征描述','计量单位','工程量','金额','综合单价','合价','col8','col7'}
                if t.replace(' ', '').lower() in {'col8','col7'}:
                    return False
                if t in bad_tokens:
                    return False
                if t in {'合计','本页小计','小计'}:
                    return False
                return True

            # 非空比例
            total_cells, non_empty_cells = 0, 0
            header_canon = [clean_cell(h) for h in header]
            header_tokens = set([re.sub(r"[\s|]", "", h) for h in header_canon if h])
            header_like_rows = 0
            effective_rows = 0
            feature_effective_rows = 0
            summary_only_rows = 0

            for row in body:
                cells = [clean_cell(x) for x in row]
                # 统计非空
                total_cells += len(cells)
                non_empty_cells += sum(1 for c in cells if c.strip())

                # 行是否像重复表头
                row_token_set = set([re.sub(r"[\s|]", "", c) for c in cells if c])
                if header_tokens and len(row_token_set & header_tokens) >= max(3, int(0.6 * len(header_tokens))):
                    header_like_rows += 1
                    continue

                # 汇总行
                if any(tok in cells for tok in ['合计', '本页小计', '小计']):
                    summary_only_rows += 1
                    continue

                # 关键列有效性
                def cell_has_effective(idx_list: List[int]) -> bool:
                    for i in idx_list:
                        if i < len(cells) and is_effective_text(cells[i]):
                            return True
                    return False

                row_effective = any(is_effective_text(c) for c in cells)
                if row_effective:
                    effective_rows += 1
                if idx_feature and cell_has_effective(idx_feature):
                    feature_effective_rows += 1

            # 判定规则
            if total_cells == 0:
                return True
            non_empty_ratio = non_empty_cells / max(1, total_cells)
            if non_empty_ratio < 0.05:
                return True
            # 没有任何有效数据行
            if effective_rows == 0:
                return True
            # 有 feature_desc 列但没有有效内容
            if idx_feature and feature_effective_rows == 0:
                return True
            # 正文几乎都是表头重复或汇总
            body_rows = len(body)
            if body_rows > 0 and (header_like_rows + summary_only_rows) >= body_rows:
                return True
            return False
        except Exception:
            # 保守回退：出错时不过滤
            return False

    def _create_desc_chunk(self, page, page_blocks: List[Any], table_bbox: Optional[List[float]],
                           page_idx: int, chunk_index: int, construction_stage: str,
                           document_id: int, ref_table_chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """基于表格附近文本，抽取“描述块”。"""
        if not page_blocks or not table_bbox:
            return None

        x0_t, y0_t, x1_t, y1_t = table_bbox

        def block_to_text(b):
            try:
                return str(b[4])
            except Exception:
                return ""

        upper, lower = [], []
        for b in page_blocks:
            if not isinstance(b, (list, tuple)) or len(b) < 5:
                continue
            x0, y0, x1, y1, txt = b[:5]
            if not txt or not str(txt).strip():
                continue
            # 重叠过大的块跳过（可能是表格内文本）
            if self._iou((x0, y0, x1, y1), (x0_t, y0_t, x1_t, y1_t)) > 0.2:
                continue
            if y1 <= y0_t:
                upper.append((y1, block_to_text(b)))
            elif y0 >= y1_t:
                lower.append((y0, block_to_text(b)))

        def select_blocks(candidates: List[Tuple[float, str]], reverse_sort: bool) -> List[str]:
            if not candidates:
                return []
            candidates = sorted(candidates, key=lambda x: x[0], reverse=reverse_sort)
            selected, total_len = [], 0
            key_terms = ["说明", "注", "适用范围", "定义", "条件", "备注"]
            for _, t in candidates:
                t_clean = str(t).strip()
                if not t_clean:
                    continue
                score = 1
                if any(k in t_clean for k in key_terms):
                    score += 2
                if total_len + len(t_clean) > 1200:
                    break
                selected.append((score, t_clean))
                total_len += len(t_clean)
            selected = [t for _, t in sorted(selected, key=lambda x: x[0], reverse=True)]
            return selected[:5]

        upper_sel = select_blocks(upper, reverse_sort=True)
        lower_sel = select_blocks(lower, reverse_sort=False)
        texts = upper_sel + lower_sel
        if not texts:
            return None

        desc_text = "\n".join(texts)
        if len(desc_text) < 50:
            return None

        title = f"## 描述 - 第{page_idx+1}页"
        if construction_stage:
            title = f"## 【{construction_stage}】描述 - 第{page_idx+1}页"
        content_md = f"{title}\n\n{desc_text}"
        full_text = (f"【{construction_stage}】\n" if construction_stage else "") + desc_text

        return {
            "construction_stage": construction_stage,
            "content": content_md,
            "text": full_text,
            "type": "desc",
            "page_idx": page_idx,
            "bbox": [],
            "chunk_index": chunk_index,
            "ref_table_chunk_index": ref_table_chunk.get("chunk_index"),
            "table_group_id": ref_table_chunk.get("table_group_id"),
            "metadata": {
                "source": "pymupdf",
                "is_table": False,
                "text_length": len(full_text),
                "construction_stage": construction_stage,
                "document_id": str(document_id)
            }
        }

    def _extract_desc_from_page(self, page, page_idx: int, chunk_index: int, document_id: int) -> Optional[Dict[str, Any]]:
        """针对整页尝试抽取描述块（非表格页召回）。"""
        try:
            blocks = page.get_text("blocks") or []
        except Exception:
            blocks = []
        if not blocks:
            return None

        # 关键词触发（工程量表相关 + 描述词）
        key_terms = [
            "分部分项工程量清单与计价表", "工程量清单", "计价表", "项目特征描述", "特征描述",
            "计量单位", "工程量", "工程名称", "项目名称", "说明", "注", "适用范围", "定义", "条件", "备注"
        ]
        texts = []
        total_len = 0
        for b in blocks:
            if not isinstance(b, (list, tuple)) or len(b) < 5:
                continue
            txt = str(b[4]).strip()
            if not txt:
                continue
            if any(k in txt for k in key_terms):
                if total_len + len(txt) > max(1200, self.min_chunk_size * 12):
                    break
                texts.append(txt)
                total_len += len(txt)

        if not texts:
            return None
        full = "\n".join(texts)
        if len(full) < max(50, self.min_chunk_size // 2):
            return None

        title = f"## 描述 - 第{page_idx+1}页"
        return {
            "construction_stage": "",
            "content": f"{title}\n\n{full}",
            "text": full,
            "type": "desc",
            "page_idx": page_idx,
            "bbox": [],
            "chunk_index": chunk_index,
            "metadata": {
                "source": "pymupdf",
                "is_table": False,
                "text_length": len(full),
                "document_id": str(document_id)
            }
        }

    def _iou(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        """计算两个矩形框的 IoU（用于排除与表格重叠的文本块）。"""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
        inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
        inter_w, inter_h = max(0.0, inter_x1 - inter_x0), max(0.0, inter_y1 - inter_y0)
        inter = inter_w * inter_h
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        union = area_a + area_b - inter if area_a + area_b - inter > 0 else 1.0
        return inter / union

    # ========= 表格 Markdown 清理 =========
    def _table_markdown_clean(self, table) -> str:
        """生成表格 Markdown，并删除全空单元格的行，减少无效 token。"""
        try:
            md = table.to_markdown()
        except Exception:
            return ""

        def is_header_sep(s: str) -> bool:
            # 匹配 |---| 或 |:---:| 等分隔线
            return re.match(r'^\|\s*:?[-]{3,}\s*(\|\s*:?[-]{3,}\s*)+\|?$', s) is not None

        def cell_text_empty(p: str) -> bool:
            # 去除 <br> 等简单换行标签后判空
            t = re.sub(r'(<br\s*/?>)+', '', p, flags=re.I).strip()
            t = t.strip('-').strip('|').strip()
            return t == ''

        lines = [l for l in (md.splitlines() if md else []) if l is not None and l.strip() != ""]
        new_lines = []
        for l in lines:
            s = l.strip()
            if not s.startswith('|'):
                # 不是表格行，原样保留
                new_lines.append(l)
                continue
            if is_header_sep(s):
                new_lines.append(l)
                continue
            # 普通表格行：判断是否全空
            parts = [p.strip() for p in s.strip('|').split('|')]
            if len(parts) == 0:
                continue
            if all(cell_text_empty(p) for p in parts):
                # 丢弃全空行（如 ||||||||）
                continue
            new_lines.append(l)

        return "\n".join(new_lines)


# ========== CLI：预览与导出 ==========
def _canonical(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\s：:、，,。;；()（）\-]+", "", s)
    return s

def _load_factor_names(factors_json_path: Optional[Path]) -> List[str]:
    names = []
    if not factors_json_path:
        return names
    try:
        with open(factors_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for stage in data:
            for src in stage.get("sources", []):
                for fac in src.get("factors", []):
                    nm = fac.get("factor")
                    if nm:
                        names.append(nm)
    except Exception:
        pass
    # 去重
    can = {}
    for n in names:
        can[_canonical(n)] = n
    return list(can.values())

def _match_factors(text: str, factor_names: List[str]) -> List[str]:
    if not text or not factor_names:
        return []
    t = _canonical(text)
    hits = []
    for name in factor_names:
        if _canonical(name) and _canonical(name) in t:
            hits.append(name)
    # 去重保序
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out

def _stable_chunk_id(file_path: str, page_idx: int, chunk_index: int, text: str) -> str:
    seed = f"{file_path}|{page_idx}|{chunk_index}|{len(text)}"
    return hashlib.md5(seed.encode('utf-8')).hexdigest()[:16]

def run_cli():
    parser = argparse.ArgumentParser(description="轻量分块：预览与导出 CSV")
    parser.add_argument("--input", type=str, required=True, help="输入 PDF 文件或目录")
    parser.add_argument("--save-csv", type=str, help="导出 CSV 路径")
    parser.add_argument("--preview", type=int, default=10, help="预览前 N 条（默认10）")
    parser.add_argument("--factors-json", type=str, help="因子 JSON 路径（可选，用于命中统计）")
    parser.add_argument("--max-pages", type=int, help="最多处理页数")
    parser.add_argument("--enable-ocr", action='store_true', help="启用 OCR（默认关闭）")
    parser.add_argument("--no-ocr", action='store_true', help="禁用 OCR（与 --enable-ocr 互斥，默认禁用）")
    parser.add_argument("--no-require-feature", action='store_true', help="不强制要求表头包含'项目特征描述'列/文本包含关键词（默认强制）")
    parser.add_argument("--force-ocr", action='store_true', help="无论是否为扫描版，都强制使用 OCR 处理")
    parser.add_argument("--min-chars", type=int, default=50, help="最小分块字符数阈值（默认50）")
    # 过滤相关
    parser.add_argument("--filter-measure", action='store_true', help="仅保留与'分部分项工程量清单与计价表/工程量清单/计价表'相关的块")
    parser.add_argument("--require-feature-strict", action='store_true', default=True, help="严格要求含'项目特征描述/特征描述'（默认开启）")
    parser.add_argument("--only-table", action='store_true', default=True, help="仅保留表格块（默认开启）")
    parser.add_argument("--preview-chars", type=int, default=400, help="日志与CSV的文本预览长度（默认400）")
    parser.add_argument("--include-table-md", action='store_true', default=True, help="在CSV中包含 table_markdown 原文（默认开启，仅table块有效）")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"输入不存在: {inp}")

    factors_json_path = Path(args.factors_json) if args.factors_json else None
    factor_names = _load_factor_names(factors_json_path)

    enable_ocr = False
    if args.enable_ocr:
        enable_ocr = True
    if args.no_ocr:
        enable_ocr = False

    chunker = LightweightChunker(
        enable_ocr=enable_ocr,
        min_chunk_size=max(1, args.min_chars),
        extract_tables_only=True,
        max_pages=args.max_pages,
        require_feature_col=(not args.no_require_feature),
        force_ocr=args.force_ocr,
        enable_desc_from_non_table=False
    )

    all_rows: List[Dict[str, Any]] = []

    # 过滤函数
    measure_kw = [
        "分部分项工程量清单与计价表", "工程量清单", "计价表",
        # 常见变体
        "分部分项工程量清单", "工程量 计价表", "清单与计价表"
    ]
    feature_kw = ["项目特征描述", "特征描述"]

    def is_measure_related(text: str) -> bool:
        if not text:
            return False
        return any(k in text for k in measure_kw)

    def has_feature_desc(headers_norm: List[str], text: str) -> bool:
        if headers_norm and any(h == "feature_desc" for h in headers_norm):
            return True
        if text and any(k in text for k in feature_kw):
            return True
        return False

    def handle_file(pdf_path: Path):
        try:
            chunks = chunker.process_pdf_file(str(pdf_path), document_id=0)
        except Exception as e:
            print(f"处理失败: {pdf_path.name} - {e}")
            return
        for c in chunks:
            block_type = c.get("type", "table")
            headers_norm = c.get("headers_norm", [])
            if block_type == "table":
                src_text = c.get("table_markdown") or c.get("text", "")
            else:
                src_text = c.get("text", "")
            factor_refs = _match_factors(src_text, factor_names) if factor_names else []

            # 精确过滤（如启用）
            if args.filter_measure and not is_measure_related(src_text):
                continue
            if args.require_feature_strict and not has_feature_desc(headers_norm, src_text):
                continue
            if args.only_table and block_type != "table":
                continue

            row = {
                "chunk_id": _stable_chunk_id(str(pdf_path), c.get("page_idx", -1), c.get("chunk_index", -1), c.get("text", "")),
                "file": pdf_path.name,
                "page_idx": c.get("page_idx", -1),
                "block_type": block_type,
                "stage": c.get("construction_stage", ""),
                "headers_norm": "|".join(headers_norm) if headers_norm else "",
                "table_group_id": c.get("table_group_id", ""),
                "factor_refs": json.dumps(factor_refs, ensure_ascii=False),
                "text_preview": (c.get("text", "")[: args.preview_chars] + ("..." if len(c.get("text", "")) > args.preview_chars else ""))
            }
            if args.include_table_md:
                row["table_markdown"] = c.get("table_markdown", "") if block_type == "table" else ""
            all_rows.append(row)

    if inp.is_dir():
        for pdf in sorted(inp.glob("*.pdf")):
            handle_file(pdf)
    else:
        handle_file(inp)

    # 预览
    print(f"共生成 {len(all_rows)} 条 chunks，预览前 {min(args.preview, len(all_rows))} 条：")
    for i, r in enumerate(all_rows[: args.preview]):
        print(f"[{i+1}] {r['block_type']}")
        print(f"  file={r['file']} page={r['page_idx']} stage={r['stage']} group={r['table_group_id']}")
        print(f"  headers={r['headers_norm']}")
        print(f"  factors={r['factor_refs']}")
        print(f"  text={r['text_preview']}")

    # 导出 CSV
    if args.save_csv:
        outp = Path(args.save_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [
                "chunk_id","file","page_idx","block_type","stage","headers_norm","table_group_id","factor_refs","text_preview"
            ])
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
        print(f"CSV 已保存：{outp}")


if __name__ == "__main__":
    run_cli()


__all__ = ['LightweightChunker']

