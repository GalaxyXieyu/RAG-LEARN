"""
配置文件
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
FINTURN_ROOT = Path(__file__).parent

# 数据目录
DATA_DIR = FINTURN_ROOT / "data"
REPORTS_DIR = DATA_DIR / "report"
OCR_CHUNKS_DIR = DATA_DIR / "ocr_chunks"
RETRIEVAL_RESULTS_DIR = DATA_DIR / "retrieval_results"

# 源文件目录
SOURCE_TMP_DIR = PROJECT_ROOT / "storage" / "tmp"

# 创建数据目录
for dir_path in [REPORTS_DIR, OCR_CHUNKS_DIR, RETRIEVAL_RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Embedding API 配置（已弃用，使用 PipelineConfig）
EMBEDDING_API_CONFIG = {
    "model_name": "BAAI/bge-m3",
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": os.getenv("SILICONFLOW_API_KEY", "sk-wjtbsdylwvtfrhpufvnpnimimatweskfemuqomzcpaftobgc")
}

# Milvus 配置
MILVUS_DB = "SZAI"
MILVUS_COLLECTION = "projects_documents_chunks"

# 因子 JSON 路径
DEFAULT_FACTORS_JSON = PROJECT_ROOT / "app" / "services" / "analysis_v2" / "factors_service" / "default_factors_scope.json"

# OCR 配置（已弃用，使用 PipelineConfig）
OCR_CONFIG = {
    "enable_ocr": True,
    "min_chunk_size": 100,
    "extract_tables_only": True
}


# ============ 新的统一配置类 ============

@dataclass
class PipelineConfig:
    """统一的流水线配置类"""
    
    # Embedding 模型配置
    model_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "EMBEDDING_MODEL_DIR",
                "/data/xieyu/Teaching/RAG/09_Embedding_Finetune/Qwen/Qwen3-Embedding-0.6B"
            )
        )
    )
    cuda_device: int = field(
        default_factory=lambda: int(os.getenv("CUDA_DEVICE", "3"))
    )
    batch_size: int = 32
    insert_batch_size: int = 1000
    
    # LLM 配置（用于 stage 分类和打标）
    llm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL",
            "https://llm.3qiao.vip:23436/v1"
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv(
            "LLM_API_KEY",
            "sk-T3bQTqP2jlTMzjXJqjf9j4rnSuxxLmzH6EFGMN3afEYG2pLi"
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL",
            "qwen2.5-72b-instruct-awq"
        )
    )
    llm_timeout: int = 20
    llm_retries: int = 2
    llm_concurrency: int = 5
    
    # Milvus 配置
    milvus_host: str = "127.0.0.1"
    milvus_port: str = "19530"
    milvus_db: str = MILVUS_DB
    default_collection: str = "projects_documents_chunks_v2"
    
    # 数据目录配置
    reports_dir: Path = REPORTS_DIR
    chunks_dir: Path = OCR_CHUNKS_DIR
    retrieval_results_dir: Path = RETRIEVAL_RESULTS_DIR
    source_tmp_dir: Path = SOURCE_TMP_DIR
    
    # Chunking 配置
    enable_ocr: bool = True
    min_chunk_size: int = 100
    extract_tables_only: bool = True
    require_feature_col: bool = True
    force_ocr: bool = False
    enable_desc_from_non_table: bool = False
    
    # 表格合并配置
    enable_table_merge: bool = True
    table_merge_max_gap: int = 1
    
    # Stage 分类配置
    enable_stage_classification: bool = True
    stage_use_llm: bool = True  # 是否使用 LLM 分类（否则返回默认值）
    
    # 检索配置
    retrieval_limit: int = 10
    enable_labeling: bool = True
    use_content_field: bool = True
    query_with_stage: bool = False
    
    def __post_init__(self):
        """验证配置"""
        # 确保必要的目录存在
        for dir_path in [self.reports_dir, self.chunks_dir, self.retrieval_results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 验证模型目录
        if not self.model_dir.exists():
            raise ValueError(f"Embedding 模型目录不存在: {self.model_dir}")
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """从环境变量创建配置（使用默认值 + 环境变量覆盖）"""
        return cls()
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "model_dir": str(self.model_dir),
            "cuda_device": self.cuda_device,
            "batch_size": self.batch_size,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "default_collection": self.default_collection,
            "enable_table_merge": self.enable_table_merge,
            "stage_use_llm": self.stage_use_llm,
}

