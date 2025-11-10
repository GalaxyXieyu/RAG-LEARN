"""
批处理阶段操作封装（重构版）：
- phase1_generate_chunks_json: 扫描 PDF → 生成 chunks JSON（集成表格合并）
- phase2_batch_ingest_to_milvus: 从 chunks JSON → 向量 → 入库（纯 LLM stage 分类）
- phase3_retrieval_and_export: 因子检索 + 打标 + 导出
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from pymilvus import connections, Collection
from finturn_embeding.config import PipelineConfig
from finturn_embeding.services import PDFChunker, merge_table_groups
from finturn_embeding.utils.embedder import get_embedder
from finturn_embeding.utils.chunks_io import lookup_content
from finturn_embeding.utils.stage import StageClassifier, LLMConfig
from finturn_embeding.vector_db.milvus_adapter import MilvusAdapter
from finturn_embeding.retrieval.retrieval_and_label import test_retrieval
from scipy.sparse import csr_matrix

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


def phase1_generate_chunks_json(
    reports_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    阶段1：生成 chunks JSON（集成表格合并）
    
    Args:
        reports_dir: PDF 根目录（默认使用配置）
        output_dir: chunks JSON 输出目录（默认使用配置）
        config: 流水线配置（默认从环境变量创建）
    
    Returns:
        处理结果统计
    """
    # 使用配置
    if config is None:
        config = PipelineConfig.from_env()
    
    reports_dir = Path(reports_dir) if reports_dir else config.reports_dir
    output_dir = Path(output_dir) if output_dir else config.chunks_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 扫描 PDF 文件
    pdf_files = sorted(list(reports_dir.rglob("*.pdf")))
    total_pdfs = len(pdf_files)
    print(f"找到 {total_pdfs} 个 PDF 文件")
    
    if total_pdfs == 0:
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    # 创建 chunker
    chunker = PDFChunker(
        enable_ocr=config.enable_ocr,
        min_chunk_size=config.min_chunk_size,
        extract_tables_only=config.extract_tables_only,
        require_feature_col=config.require_feature_col,
        force_ocr=config.force_ocr,
        enable_desc_from_non_table=config.enable_desc_from_non_table,
    )
    
    success = failed = skipped = 0
    
    for pdf in tqdm(pdf_files, desc="Generating chunks JSON", unit="file"):
        document_id = pdf.parent.name
        out_json = output_dir / f"{document_id}.json"
        
        # 跳过已处理
        if out_json.exists():
            skipped += 1
            continue
        
        try:
            # 1. 处理 PDF
            chunks = chunker.process_pdf(pdf, document_id=0)
            
            if not chunks:
                failed += 1
                continue
            
            # 2. 表格合并（如果启用）
            if config.enable_table_merge:
                chunks = merge_table_groups(chunks, max_gap=config.table_merge_max_gap)
            
            # 3. 提取 markdown 表格
            markdown_tables = chunker._extract_markdown_tables(chunks)
            
            # 4. 保存结果
            data = {
                "document_id": document_id,
                "source_file": pdf.name,
                "chunks": chunks,
                "markdown_tables": markdown_tables,
                "total_chunks": len(chunks),
                "total_tables": len(markdown_tables),
                "table_merged": config.enable_table_merge
            }
            
            out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            success += 1
            
        except Exception as e:
            tqdm.write(f"  ❌ {document_id}: 解析失败 - {e}")
            failed += 1
    
    return {
        "total": total_pdfs,
        "success": success,
        "failed": failed,
        "skipped": skipped
    }


def phase2_batch_ingest_to_milvus(
    chunks_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    clear_collection: bool = False,
) -> Dict[str, Any]:
    """
    阶段2：批量入库到 Milvus（纯 LLM stage 分类）
    
    Args:
        chunks_dir: chunks JSON 目录（默认使用配置）
        collection_name: 集合名称（默认使用配置）
        config: 流水线配置（默认从环境变量创建）
        clear_collection: 是否清空集合
    
    Returns:
        处理结果统计
    """
    print("\n【第二阶段】批量入库到 Milvus（纯 LLM stage 分类）")
    
    # 使用配置
    if config is None:
        config = PipelineConfig.from_env()
    
    chunks_dir = Path(chunks_dir) if chunks_dir else config.chunks_dir
    collection_name = collection_name or config.default_collection
    
    # 连接 Milvus
    try:
        adapter = MilvusAdapter()
        col = adapter.get_collection(collection_name)
        print(f"✅ Milvus 连接成功，集合: {collection_name}")
    except Exception as e:
        print(f"❌ Milvus 连接失败: {e}")
        return {"total": 0, "success": 0, "failed": 0}
    
    # 清空集合（如果需要）
    if clear_collection:
        try:
            col.delete(expr="chunk_id != ''")
            print("✅ 集合已清空")
        except Exception as e:
            print(f"⚠️ 清空集合失败: {e}")
    
    # 加载 Embedding 模型
    try:
        embedder = get_embedder(config.model_dir, cuda_device=config.cuda_device)
        print(f"✅ Embedding 模型加载成功 (cuda_device={config.cuda_device})")
    except Exception as e:
        print(f"❌ Embedding 模型加载失败: {e}")
        return {"total": 0, "success": 0, "failed": 0}
    
    # 创建 Stage 分类器（纯 LLM）
    if config.enable_stage_classification and config.stage_use_llm:
        clf = StageClassifier(
            LLMConfig(
                base_url=config.llm_base_url,
                api_key=config.llm_api_key,
                model=config.llm_model,
                timeout=config.llm_timeout,
                retries=config.llm_retries,
            )
        )
        print("✅ 启用纯 LLM stage 分类")
    else:
        clf = StageClassifier(None)  # 返回默认值
        print("⚠️ Stage 分类已禁用，使用默认值")
    
    # 扫描 chunks 文件
    chunk_files = sorted(list(chunks_dir.glob("*.json")))
    total_files = len(chunk_files)
    total_chunks_inserted = 0
    success = failed = 0
    
    for chunk_file in tqdm(chunk_files, desc="Ingesting to Milvus", unit="file"):
        try:
            # 加载数据
            data = json.loads(chunk_file.read_text(encoding="utf-8"))
            chunks = data.get("chunks", [])
            document_id = data.get("document_id", chunk_file.stem)
            source_file = data.get("source_file", "")
            
            if not chunks:
                continue
            
            # 准备数据
            texts: List[str] = []
            rows: List[Dict[str, Any]] = []
            
            for i, c in enumerate(chunks):
                block_type = c.get("type", "table")
                headers_norm = "|".join(c.get("headers_norm", []) or [])
                table_group_id = c.get("table_group_id", "")
                page_idx = int(c.get("page_idx", -1))
                txt = c.get("text", "")
                
                if not txt.strip():
                    continue
                
                # Stage 分类（纯 LLM）
                stage = clf.classify_from_chunk(c)
                
                chunk_index = int(c.get("chunk_index", i))
                chunk_id = f"{document_id}-{page_idx}-{chunk_index}"[:64]
                
                rows.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id[:256],
                    "source_file": source_file[:512],
                    "project_code": "",
                    "page_idx": page_idx,
                    "block_type": block_type[:32],
                    "stage": stage[:32],
                    "table_group_id": table_group_id[:64],
                    "headers_norm": headers_norm[:512],
                    "text": txt[:8192],
                })
                texts.append(txt)
            
            if not rows:
                continue
            
            # 生成向量
            dense_vecs = embedder.encode(texts, batch_size=config.batch_size)
            
            # 批量插入
            total_inserted = 0
            num_batches = (len(rows) + config.insert_batch_size - 1) // config.insert_batch_size
            
            for b in range(num_batches):
                s, e = b * config.insert_batch_size, min((b + 1) * config.insert_batch_size, len(rows))
                batch_rows = rows[s:e]
                batch_vecs = dense_vecs[s:e]
                
                # 插入：如果集合 schema 有 sparse_vector 字段，需要传占位符
                # 否则传 None
                try:
                    _n = adapter.insert_rows(collection_name, batch_rows, batch_vecs)
                    total_inserted += _n
                except Exception as e:
                    # 如果失败且是字段不匹配，尝试传入占位稀疏向量
                    if "doesn't match with schema fields" in str(e):
                        sparse_vecs = [csr_matrix(([0.001], ([0], [0])), shape=(1, 1)) for _ in batch_rows]
                        _n = adapter.insert_rows(collection_name, batch_rows, batch_vecs, sparse_vecs=sparse_vecs)
                        total_inserted += _n
                    else:
                        raise
            
            if total_inserted:
                total_chunks_inserted += total_inserted
                success += 1
                
        except Exception as e:
            tqdm.write(f"  ❌ {chunk_file.name}: 处理失败 - {e}")
            failed += 1
    
    print(f"总共入库 chunks: {total_chunks_inserted}")
    return {
        "total": total_files,
        "success": success,
        "failed": failed,
        "total_chunks_inserted": total_chunks_inserted
    }


def phase3_retrieval_and_export(
    collection_name: Optional[str] = None,
    queries: Optional[List[str]] = None,
    config: Optional[PipelineConfig] = None,
    stage_filter: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    阶段3：因子检索 + 打标 + 导出
    
    Args:
        collection_name: 集合名称（默认使用配置）
        queries: 查询词列表
        config: 流水线配置（默认从环境变量创建）
        stage_filter: 阶段筛选
        output_dir: 导出目录（默认使用配置）
    
    Returns:
        处理结果
    """
    if config is None:
        config = PipelineConfig.from_env()
    
    collection_name = collection_name or config.default_collection
    queries = queries or []
    
    if not queries:
        print("⚠️ 未提供查询词，跳过检索")
        return {"total": 0, "success": 0, "failed": 0}
    
    return test_retrieval(
        collection_name=collection_name,
        model_dir=config.model_dir,
        queries=queries,
        cuda_device=config.cuda_device,
        limit=config.retrieval_limit,
        expr=None,
        output_dir=output_dir or config.retrieval_results_dir,
        enable_labeling=config.enable_labeling,
        llm_base_url=config.llm_base_url,
        llm_api_key=config.llm_api_key,
        llm_model=config.llm_model,
        llm_concurrency=config.llm_concurrency,
        chunks_dir=config.chunks_dir,
        use_content_field=config.use_content_field,
        query_with_stage=config.query_with_stage,
        stage_filter=stage_filter,
    )
