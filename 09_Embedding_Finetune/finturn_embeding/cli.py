"""
统一 CLI：将各步骤功能以子命令形式提供（重构版）

使用新的配置类和 services

子命令：
- chunks   ：扫描 PDF → 生成 chunks JSON（phase1，集成表格合并）
- ingest   ：从 chunks JSON → 向量 → 入库（phase2，纯 LLM stage 分类）
- retrieve ：检索 + LLM 打标（按查询词）
- pipeline ：三阶段编排（1/2/3 或 all，核心功能）
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import argparse
import json

from .config import PipelineConfig, DEFAULT_FACTORS_JSON
from .pipelines.batch_ops import (
    phase1_generate_chunks_json,
    phase2_batch_ingest_to_milvus,
    phase3_retrieval_and_export,
)
from .retrieval.retrieval_and_label import test_retrieval as retrieval_and_label


def _extract_queries_with_stage(json_path: Path) -> List[str]:
    """
    从因子 JSON 提取所有因子，生成带 stage 的 query
    
    Args:
        json_path: 因子 JSON 文件路径
    
    Returns:
        ["{stage} {factor}", ...] 列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = []
    
    for stage_obj in data:
        stage = stage_obj.get("stage", "")
        
        for source in stage_obj.get("sources", []):
            for factor_obj in source.get("factors", []):
                factor = factor_obj.get("factor", "")
                
                if factor:
                    # 生成 query："{stage} {factor}"
                    query = f"{stage} {factor}"
                    queries.append(query)
    
    return queries


def cmd_chunks(args: argparse.Namespace) -> None:
    """子命令：生成 chunks JSON"""
    config = PipelineConfig.from_env()
    
    # 覆盖配置（如果提供了参数）
    if args.reports_dir:
        config.reports_dir = Path(args.reports_dir)
    if args.output_dir:
        config.chunks_dir = Path(args.output_dir)
    if args.enable_ocr:
        config.enable_ocr = True
    
    result = phase1_generate_chunks_json(config=config)
    print(f"\n结果: {result}")


def cmd_ingest(args: argparse.Namespace) -> None:
    """子命令：从 chunks JSON 入库 Milvus"""
    config = PipelineConfig.from_env()
    
    # 覆盖配置（如果提供了参数）
    if args.chunks_dir:
        config.chunks_dir = Path(args.chunks_dir)
    if args.model_dir:
        config.model_dir = Path(args.model_dir)
    if args.collection:
        config.default_collection = args.collection
    if args.cuda_device is not None:
        config.cuda_device = args.cuda_device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.insert_batch_size:
        config.insert_batch_size = args.insert_batch_size
    
    result = phase2_batch_ingest_to_milvus(
        config=config,
        clear_collection=args.clear_collection,
    )
    print(f"\n结果: {result}")


def cmd_retrieve(args: argparse.Namespace) -> None:
    """子命令：检索 + 打标"""
    config = PipelineConfig.from_env()
    
    # 覆盖配置
    if args.model_dir:
        config.model_dir = Path(args.model_dir)
    if args.collection:
        config.default_collection = args.collection
    if args.cuda_device is not None:
        config.cuda_device = args.cuda_device
    if args.limit:
        config.retrieval_limit = args.limit
    
    out_dir = Path(args.output_dir) if args.output_dir else config.retrieval_results_dir
    
    result = retrieval_and_label(
        collection_name=config.default_collection,
        model_dir=config.model_dir,
        queries=args.queries,
        cuda_device=config.cuda_device,
        limit=config.retrieval_limit,
        expr=None,
        output_dir=out_dir,
        enable_labeling=True,
        llm_base_url=config.llm_base_url,
        llm_api_key=config.llm_api_key,
        llm_model=config.llm_model,
        llm_concurrency=config.llm_concurrency,
        chunks_dir=config.chunks_dir,
        use_content_field=True,
        query_with_stage=False,
        stage_filter=args.stage,
    )
    print(f"\n结果: {result}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    """子命令：三阶段编排"""
    config = PipelineConfig.from_env()
    
    # 覆盖配置
    if args.model_dir:
        config.model_dir = Path(args.model_dir)
    if args.collection:
        config.default_collection = args.collection
    if args.cuda_device is not None:
        config.cuda_device = args.cuda_device
    
    # 阶段1
    if args.phase in ("1", "1-2", "all"):
        print("\n" + "=" * 60)
        print("【阶段1】生成 chunks JSON（集成跨页表格合并）")
        print("=" * 60)
        result = phase1_generate_chunks_json(config=config)
        print(f"结果: {result}")
    
    # 阶段2
    if args.phase in ("2", "1-2", "2-3", "all"):
        print("\n" + "=" * 60)
        print("【阶段2】入库到 Milvus（纯 LLM stage 分类）")
        print("=" * 60)
        result = phase2_batch_ingest_to_milvus(
            config=config, 
            clear_collection=args.clear_collection
        )
        print(f"结果: {result}")
    
    # 阶段3
    if args.phase in ("3", "2-3", "all"):
        # 提取 queries
        queries = args.queries
        
        # 如果使用 --use-all-factors，从 JSON 提取所有因子
        if args.use_all_factors:
            print("\n📖 从 default_factors_scope.json 提取所有因子...")
            factors_json = Path(args.factors_json) if args.factors_json else DEFAULT_FACTORS_JSON
            queries = _extract_queries_with_stage(factors_json)
            print(f"✅ 共提取 {len(queries)} 个因子（格式：stage + factor）")
            print(f"示例前 5 个：")
            for q in queries[:5]:
                print(f"  - {q}")
        
        if not queries:
            raise SystemExit("❌ phase 包含 3 时必须提供 --queries 或 --use-all-factors")
        
        print("\n" + "=" * 60)
        print("【阶段3】检索 + 打标 + 导出")
        print(f"Query 数量: {len(queries)}")
        print("=" * 60)
        
        out_dir = Path(args.output_dir) if args.output_dir else None
        result = phase3_retrieval_and_export(
            queries=queries,
            config=config,
            stage_filter=args.stage,
            output_dir=out_dir,
        )
        print(f"结果: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="finturn_embeding 统一 CLI（使用 PipelineConfig）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 全流程：重新 chunk + 入库 + 检索所有因子（推荐）
  python -m finturn_embeding.cli pipeline \\
    --phase all \\
    --clear-collection \\
    --use-all-factors
  
  # 仅检索阶段（使用所有因子）
  python -m finturn_embeding.cli pipeline \\
    --phase 3 \\
    --use-all-factors
  
  # 手动指定查询词
  python -m finturn_embeding.cli pipeline \\
    --phase all \\
    --queries "土石方工程 工程量" "地基处理工程 桩径"
        """
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    # chunks 子命令
    p_chunks = sub.add_parser("chunks", help="生成 chunks JSON（集成表格合并）")
    p_chunks.add_argument("--reports-dir", type=str, help="PDF 根目录（默认配置）")
    p_chunks.add_argument("--output-dir", type=str, help="chunks JSON 输出目录（默认配置）")
    p_chunks.add_argument("--enable-ocr", action="store_true", help="启用 OCR")
    p_chunks.set_defaults(func=cmd_chunks)
    
    # ingest 子命令
    p_ingest = sub.add_parser("ingest", help="从 chunks JSON 入库 Milvus（纯 LLM stage 分类）")
    p_ingest.add_argument("--chunks-dir", type=str, help="chunks JSON 目录（默认配置）")
    p_ingest.add_argument("--collection", type=str, help="集合名称（默认配置）")
    p_ingest.add_argument("--model-dir", type=str, help="本地 embedding 模型目录（默认配置）")
    p_ingest.add_argument("--cuda-device", type=int, help="CUDA 设备（默认配置）")
    p_ingest.add_argument("--batch-size", type=int, help="批次大小（默认配置）")
    p_ingest.add_argument("--insert-batch-size", type=int, help="插入批次大小（默认配置）")
    p_ingest.add_argument("--clear-collection", action="store_true", help="清空集合")
    p_ingest.set_defaults(func=cmd_ingest)
    
    # retrieve 子命令
    p_retrieve = sub.add_parser("retrieve", help="检索 + 打标")
    p_retrieve.add_argument("--queries", nargs="+", required=True, help="查询词列表")
    p_retrieve.add_argument("--collection", type=str, help="集合名称（默认配置）")
    p_retrieve.add_argument("--model-dir", type=str, help="本地 embedding 模型目录（默认配置）")
    p_retrieve.add_argument("--cuda-device", type=int, help="CUDA 设备（默认配置）")
    p_retrieve.add_argument("--limit", type=int, help="检索数量限制（默认配置）")
    p_retrieve.add_argument("--stage", type=str, help="阶段筛选")
    p_retrieve.add_argument("--output-dir", type=str, help="导出目录（默认配置）")
    p_retrieve.set_defaults(func=cmd_retrieve)
    
    # pipeline 子命令（核心功能）
    p_pipe = sub.add_parser("pipeline", help="三阶段编排（推荐：重新 chunk + 入库 + 检索所有因子）")
    p_pipe.add_argument("--phase", choices=["1", "2", "3", "1-2", "2-3", "all"], default="all",
                       help="执行阶段（默认 all）")
    p_pipe.add_argument("--queries", nargs="+", 
                       help="手动指定查询词（与 --use-all-factors 互斥）")
    p_pipe.add_argument("--use-all-factors", action="store_true",
                       help="自动从 default_factors_scope.json 提取所有因子作为 query（格式：stage + factor）")
    p_pipe.add_argument("--factors-json", type=str,
                       help="因子 JSON 路径（默认使用 default_factors_scope.json）")
    p_pipe.add_argument("--clear-collection", action="store_true",
                       help="阶段2 清空集合（重新入库时使用）")
    p_pipe.add_argument("--stage", type=str, help="阶段筛选（阶段3）")
    p_pipe.add_argument("--output-dir", type=str, help="第三阶段导出目录")
    p_pipe.add_argument("--collection", type=str, help="集合名称（默认配置）")
    p_pipe.add_argument("--model-dir", type=str, help="本地 embedding 模型目录（默认配置）")
    p_pipe.add_argument("--cuda-device", type=int, help="CUDA 设备（默认配置）")
    p_pipe.set_defaults(func=cmd_pipeline)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
