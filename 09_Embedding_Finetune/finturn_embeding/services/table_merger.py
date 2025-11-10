"""
跨页表格合并服务
基于 table_group_id 和 page_idx 合并连续的表格页
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from collections import defaultdict
import re


class TableMerger:
    """跨页表格合并器"""
    
    def __init__(self, max_gap: int = 1):
        """
        Args:
            max_gap: 允许的最大页码间隔（默认1，即只合并连续页）
        """
        self.max_gap = max_gap
    
    def merge_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并 chunks 中的跨页表格
        
        策略：
        1. 按 table_group_id 分组
        2. 在每组内，按 page_idx 排序
        3. 合并连续页（page_idx 差值 <= max_gap）
        4. 保留 desc 类型的块不合并
        
        Args:
            chunks: 原始 chunks 列表
        
        Returns:
            合并后的 chunks 列表
        """
        if not chunks:
            return []
        
        # 分离表格块和描述块
        table_chunks = [c for c in chunks if c.get("type") == "table"]
        desc_chunks = [c for c in chunks if c.get("type") == "desc"]
        
        # 按 table_group_id 分组
        groups = self._group_by_table_id(table_chunks)
        
        # 合并每组
        merged_tables = []
        for group_id, group_chunks in groups.items():
            if not group_id or not group_chunks:
                # 没有 group_id 的块不合并
                merged_tables.extend(group_chunks)
                continue
            
            # 按 page_idx 排序
            group_chunks.sort(key=lambda x: x.get("page_idx", -1))
            
            # 合并连续页
            merged = self._merge_continuous_pages(group_chunks)
            merged_tables.extend(merged)
        
        # 合并结果并重新排序
        all_chunks = merged_tables + desc_chunks
        all_chunks.sort(key=lambda x: (x.get("page_idx", -1), x.get("chunk_index", -1)))
        
        # 重新分配 chunk_index
        for i, chunk in enumerate(all_chunks):
            chunk["chunk_index"] = i
        
        return all_chunks
    
    def _group_by_table_id(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """按 table_group_id 分组"""
        groups = defaultdict(list)
        
        for chunk in chunks:
            group_id = chunk.get("table_group_id", "")
            groups[group_id].append(chunk)
        
        return dict(groups)
    
    def _merge_continuous_pages(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并连续页的表格"""
        if len(chunks) <= 1:
            return chunks
        
        merged_results = []
        current_group = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_page = chunks[i-1].get("page_idx", -1)
            curr_page = chunks[i].get("page_idx", -1)
            
            # 检查是否连续
            if 0 < (curr_page - prev_page) <= self.max_gap:
                current_group.append(chunks[i])
            else:
                # 合并当前组
                merged_results.append(self._merge_group(current_group))
                # 开始新组
                current_group = [chunks[i]]
        
        # 合并最后一组
        if current_group:
            merged_results.append(self._merge_group(current_group))
        
        return merged_results
    
    def _merge_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并一组表格块"""
        if len(group) == 1:
            return group[0]
        
        # 使用第一个块作为基础
        merged = group[0].copy()
        
        # 合并文本
        texts = []
        contents = []
        table_markdowns = []
        
        for chunk in group:
            # 收集文本
            text = chunk.get("text", "")
            if text:
                # 移除重复的阶段标记（如 【土石方工程】）
                text = self._clean_stage_prefix(text)
                texts.append(text)
            
            # 收集 content
            content = chunk.get("content", "")
            if content:
                content = self._clean_markdown_header(content)
                contents.append(content)
            
            # 收集 table_markdown
            md = chunk.get("table_markdown", "")
            if md:
                # 移除表头分隔线（避免重复）
                md = self._remove_duplicate_header_sep(md)
                table_markdowns.append(md)
        
        # 更新合并后的字段
        stage = merged.get("construction_stage", "")
        
        # 合并 text（用于向量化）
        merged_text_parts = []
        if stage:
            merged_text_parts.append(f"【{stage}】")
        page_start = group[0].get("page_idx", -1) + 1
        page_end = group[-1].get("page_idx", -1) + 1
        merged_text_parts.append(f"第{page_start}-{page_end}页表格（跨页合并）")
        merged_text_parts.extend(texts)
        merged["text"] = "\n".join(merged_text_parts)
        
        # 合并 content（用于展示）
        merged_content_parts = []
        if stage:
            merged_content_parts.append(f"## 【{stage} - 第{page_start}-{page_end}页】（跨页合并）\n")
        else:
            merged_content_parts.append(f"## 第{page_start}-{page_end}页表格（跨页合并）\n")
        merged_content_parts.extend(contents)
        merged["content"] = "\n\n".join(merged_content_parts)
        
        # 合并 table_markdown
        if table_markdowns:
            merged["table_markdown"] = "\n\n".join(table_markdowns)
        
        # 更新元数据
        metadata = merged.get("metadata", {})
        metadata["is_merged"] = True
        metadata["merged_pages"] = [c.get("page_idx", -1) for c in group]
        metadata["merged_count"] = len(group)
        metadata["text_length"] = len(merged["text"])
        
        # 更新表格行列数（累加）
        total_rows = sum(c.get("metadata", {}).get("table_rows", 0) for c in group)
        metadata["table_rows"] = total_rows
        
        merged["metadata"] = metadata
        
        # 更新 bbox（使用第一个和最后一个的合并）
        if group[0].get("bbox") and group[-1].get("bbox"):
            bbox1 = group[0]["bbox"]
            bbox2 = group[-1]["bbox"]
            # 简单合并：使用最小和最大坐标
            if len(bbox1) >= 4 and len(bbox2) >= 4:
                merged["bbox"] = [
                    min(bbox1[0], bbox2[0]),
                    min(bbox1[1], bbox2[1]),
                    max(bbox1[2], bbox2[2]),
                    max(bbox1[3], bbox2[3]),
                ]
        
        return merged
    
    def _clean_stage_prefix(self, text: str) -> str:
        """移除文本开头的阶段标记"""
        # 移除 "【xxx】" 开头
        text = re.sub(r'^【[^】]+】\s*', '', text)
        # 移除 "第N页表格" 开头
        text = re.sub(r'^第\d+页表格\s*', '', text)
        return text.strip()
    
    def _clean_markdown_header(self, content: str) -> str:
        """移除 markdown 中的标题行"""
        lines = content.split('\n')
        # 跳过以 ## 开头的标题行
        cleaned_lines = [l for l in lines if not l.strip().startswith('##')]
        return '\n'.join(cleaned_lines).strip()
    
    def _remove_duplicate_header_sep(self, markdown: str) -> str:
        """移除重复的表头分隔线"""
        lines = markdown.split('\n')
        result = []
        seen_sep = False
        
        for line in lines:
            # 检查是否是分隔线 (|---|---|)
            if re.match(r'^\|\s*:?[-]{3,}\s*(\|\s*:?[-]{3,}\s*)+\|?$', line.strip()):
                if seen_sep:
                    continue  # 跳过重复的分隔线
                seen_sep = True
            result.append(line)
        
        return '\n'.join(result)


def merge_table_groups(chunks: List[Dict[str, Any]], max_gap: int = 1) -> List[Dict[str, Any]]:
    """
    快捷函数：合并跨页表格
    
    Args:
        chunks: 原始 chunks 列表
        max_gap: 允许的最大页码间隔
    
    Returns:
        合并后的 chunks 列表
    """
    merger = TableMerger(max_gap=max_gap)
    return merger.merge_chunks(chunks)

