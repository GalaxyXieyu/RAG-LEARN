"""
报告文件获取服务
从 step1_fetch_reports.py 抽取的核心逻辑
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class ReportFetcher:
    """报告文件获取器"""
    
    def __init__(self, source_dir: Path, target_dir: Path):
        """
        Args:
            source_dir: 源目录（storage/tmp）
            target_dir: 目标目录（data/report）
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_reports(
        self,
        report_file_ids: Optional[Dict[int, Dict]] = None,
    ) -> Dict[str, Any]:
        """
        从源目录复制报告文件到目标目录
        
        Args:
            report_file_ids: 文件ID到文件信息的映射 {file_id: file_info}
                           如果为 None，则扫描源目录下所有 PDF
        
        Returns:
            处理结果统计和文件清单
        """
        if not self.source_dir.exists():
            return {
                "success": 0,
                "failed": 0,
                "skipped": 0,
                "files": [],
                "error": f"源目录不存在: {self.source_dir}"
            }
        
        # 加载已处理文件
        manifest_path = self.target_dir / "manifest.json"
        processed_files = self._load_processed_files(manifest_path)
        
        # 扫描源目录
        pdf_files = self._scan_source_directory(report_file_ids)
        
        if not pdf_files:
            return {
                "success": 0,
                "failed": 0,
                "skipped": 0,
                "files": [],
                "message": "未找到符合条件的 PDF 文件"
            }
        
        # 处理文件
        results = self._process_files(pdf_files, processed_files)
        
        # 保存 manifest
        self._save_manifest(manifest_path, results["files"])
        
        return results
    
    def _load_processed_files(self, manifest_path: Path) -> set:
        """加载已处理的文件ID"""
        processed = set()
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    processed = {f["file_id"] for f in data.get("files", [])}
            except Exception:
                pass
        return processed
    
    def _scan_source_directory(
        self,
        report_file_ids: Optional[Dict[int, Dict]] = None
    ) -> List[Dict[str, Any]]:
        """扫描源目录，查找 PDF 文件"""
        pdf_files = []
        
        for folder in self.source_dir.iterdir():
            if not folder.is_dir():
                continue
            
            # 提取 file_id
            try:
                file_id = int(folder.name)
            except ValueError:
                continue
            
            # 如果指定了文件列表，只处理列表中的文件
            if report_file_ids is not None and file_id not in report_file_ids:
                continue
            
            # 查找 PDF 文件
            for pdf_file in folder.glob("*.pdf"):
                file_info = report_file_ids.get(file_id, {}) if report_file_ids else {}
                pdf_files.append({
                    "file_id": file_id,
                    "folder_path": folder,
                    "pdf_path": pdf_file,
                    "file_name": pdf_file.name,
                    "file_info": file_info
                })
                break  # 每个文件夹只取第一个 PDF
        
        return pdf_files
    
    def _process_files(
        self,
        pdf_files: List[Dict[str, Any]],
        processed_files: set
    ) -> Dict[str, Any]:
        """处理文件复制"""
        success = 0
        failed = 0
        skipped = 0
        files = []
        
        for file_info in pdf_files:
            file_id = file_info["file_id"]
            
            # 跳过已处理
            if file_id in processed_files:
                skipped += 1
                continue
            
            # 创建目标目录
            target_dir = self.target_dir / str(file_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            target_file = target_dir / f"{file_id}_origin.pdf"
            
            try:
                shutil.copy2(file_info["pdf_path"], target_file)
                success += 1
                
                # 保存文件信息
                file_data = {
                    "file_id": file_id,
                    "file_name": file_info["file_name"],
                    "local_path": str(target_file),
                    "source_path": str(file_info["pdf_path"]),
                    "biz_type": file_info.get("file_info", {}).get("biz_type", "report"),
                    "project_code": file_info.get("file_info", {}).get("project_code", ""),
                    "origin_name": file_info.get("file_info", {}).get("origin_name", file_info["file_name"])
                }
                files.append(file_data)
                
            except Exception as e:
                failed += 1
                print(f"  复制失败 {file_id}: {e}")
        
        return {
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "files": files
        }
    
    def _save_manifest(self, manifest_path: Path, files: List[Dict]):
        """保存 manifest 文件"""
        # 加载已有文件
        existing_files = []
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing_files = data.get("files", [])
            except Exception:
                pass
        
        # 合并
        existing_ids = {f["file_id"] for f in existing_files}
        all_files = existing_files + [f for f in files if f["file_id"] not in existing_ids]
        
        # 保存
        manifest_data = {
            "total": len(all_files),
            "files": all_files
        }
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)


def fetch_reports(
    source_dir: Path,
    target_dir: Path,
    report_file_ids: Optional[Dict[int, Dict]] = None,
) -> Dict[str, Any]:
    """
    快捷函数：从源目录复制报告文件到目标目录
    
    Args:
        source_dir: 源目录
        target_dir: 目标目录
        report_file_ids: 可选的文件ID映射
    
    Returns:
        处理结果
    """
    fetcher = ReportFetcher(source_dir, target_dir)
    return fetcher.fetch_reports(report_file_ids)

