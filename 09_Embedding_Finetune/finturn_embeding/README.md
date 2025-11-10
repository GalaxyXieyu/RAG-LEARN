# Embedding 检索和 OCR 分块优化工程（重构版）

用于优化 embedding 检索和因子分块模式（OCR 模式），为后续微调 embedding 模型准备训练数据。

**✨ 重构亮点**：
- ✅ **跨页表格合并**：基于 `table_group_id` 自动合并连续页的表格
- ✅ **纯 LLM Stage 分类**：使用 LLM 精准分类工程阶段
- ✅ **纯稠密向量**：移除稀疏向量（qwen3-embedding-0.6b 不支持）
- ✅ **统一配置管理**：`PipelineConfig` 集中管理所有配置
- ✅ **原子化服务层**：`services/` 提供可复用的原子能力

---

## 📁 目录结构

```
finturn_embeding/
├── cli.py                          # 🎯 统一 CLI（唯一入口）
├── config.py                       # ⚙️ 统一配置（PipelineConfig）
│
├── services/                       # 🔧 原子能力服务层
│   ├── __init__.py
│   ├── report_fetcher.py           # 报告文件获取
│   ├── pdf_chunker.py              # PDF 分块处理
│   └── table_merger.py             # 跨页表格合并（✨新增）
│
├── pipelines/                      # 🔄 流程编排层
│   ├── batch_ops.py                # phase1/2/3 编排
│   └── retrieval_export.py         # 因子检索导出
│
├── chunking/                       # 📄 分块与 OCR
│   └── lightweight_chunker.py      # PyMuPDF 分块器
│
├── retrieval/                      # 🔍 检索与打标
│   └── retrieval_and_label.py      # 检索 + LLM 打标
│
├── vector_db/                      # 💾 向量数据库适配
│   ├── milvus_adapter.py           # Milvus 统一适配器
│   └── search.py                   # 检索封装
│
├── utils/                          # 🛠️ 工具库
│   ├── embedder.py                 # Embedding 模型加载
│   ├── llm.py                      # LLM 封装
│   ├── stage.py                    # Stage 分类（纯 LLM）
│   ├── chunks_io.py                # Chunks 反查
│   ├── db_helper.py                # 数据库辅助
│   └── file_helper.py              # 文件辅助
│
└── data/                           # 📊 数据目录
    ├── report/                     # PDF 文件
    ├── ocr_chunks/                 # Chunks JSON
    └── retrieval_results/          # 检索结果 CSV
```

---

## 🚀 快速开始

### 环境配置

```bash
# 1. 安装依赖
pip install pymupdf sentence-transformers pymilvus rapidocr-onnxruntime

# 2. 配置环境变量（可选，有默认值）
export EMBEDDING_MODEL_DIR="/path/to/Qwen3-Embedding-0.6B"
export CUDA_DEVICE="3"
export LLM_BASE_URL="https://llm.example.com/v1"
export LLM_API_KEY="your-api-key"
export LLM_MODEL="qwen2.5-72b-instruct-awq"
```

### 三阶段流程

#### 🎯 方式一：一键运行全流程

```bash
python -m finturn_embeding.cli pipeline \
  --phase all \
  --queries "工程量" "挖深" "桩径"
```

#### 📦 方式二：分步执行

**阶段 1：生成 Chunks JSON（集成跨页表格合并）**

```bash
python -m finturn_embeding.cli chunks --enable-ocr
```

- 输入：`data/report/**/*.pdf`
- 输出：`data/ocr_chunks/*.json`
- 功能：
  - PDF 分块（PyMuPDF + OCR）
  - ✨ **跨页表格自动合并**（基于 `table_group_id`）
  - 提取 "项目特征描述" 表格
  - 生成描述块（table 上下方文本）

**阶段 2：入库到 Milvus（纯 LLM Stage 分类）**

```bash
python -m finturn_embeding.cli ingest \
  --clear-collection  # 可选：清空已有数据
```

- 输入：`data/ocr_chunks/*.json`
- 输出：Milvus 集合 `projects_documents_chunks_v2`
- 功能：
  - 加载 Qwen3-Embedding-0.6B 模型
  - ✨ **纯 LLM Stage 分类**（7个工程阶段）
  - 生成稠密向量（无稀疏向量）
  - 批量入库

**阶段 3：检索 + LLM 打标 + 导出**

```bash
python -m finturn_embeding.cli retrieve \
  --queries "工程量" "挖深" "桩径" \
  --stage "土石方工程"  # 可选：按阶段筛选
```

- 输入：查询词列表
- 输出：`data/retrieval_results/doc_{document_id}_labeled.csv`
- 功能：
  - 向量检索（TopK）
  - LLM 打标（positive/negative）
  - 分文档导出 CSV

---

## 🔧 高级用法

### 因子检索（从 JSON 提取因子）

```bash
python -m finturn_embeding.cli factors \
  --factors-json /path/to/factors.json \
  --limit 10 \
  --output /tmp/factors_retrieval.csv
```

### 自定义配置

```bash
# 使用自定义模型和设备
python -m finturn_embeding.cli pipeline \
  --model-dir /path/to/custom-model \
  --cuda-device 0 \
  --collection my_custom_collection \
  --phase all \
  --queries "示例查询"
```

### 单独使用 Services（代码调用）

```python
from pathlib import Path
from finturn_embeding.services import PDFChunker, merge_table_groups
from finturn_embeding.config import PipelineConfig

# 加载配置
config = PipelineConfig.from_env()

# 1. PDF 分块
chunker = PDFChunker(
    enable_ocr=True,
    min_chunk_size=100,
    require_feature_col=True,
)
chunks = chunker.process_pdf(Path("example.pdf"), document_id=0)

# 2. 跨页表格合并
merged_chunks = merge_table_groups(chunks, max_gap=1)

# 3. 保存结果
import json
output_path = Path("output.json")
output_path.write_text(json.dumps(merged_chunks, ensure_ascii=False, indent=2))
```

---

## ⚙️ 配置说明

### PipelineConfig（统一配置类）

所有配置通过 `config.py` 中的 `PipelineConfig` 管理：

```python
from finturn_embeding.config import PipelineConfig

# 从环境变量创建（自动读取环境变量）
config = PipelineConfig.from_env()

# 或手动创建
config = PipelineConfig(
    model_dir=Path("/path/to/model"),
    cuda_device=3,
    enable_table_merge=True,      # ✨ 启用跨页表格合并
    stage_use_llm=True,            # ✨ 使用 LLM 分类 stage
    llm_base_url="https://...",
    llm_api_key="sk-...",
    llm_model="qwen2.5-72b",
)

# 查看配置
print(config.to_dict())
```

### 环境变量

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `EMBEDDING_MODEL_DIR` | Embedding 模型路径 | `/data/xieyu/.../Qwen3-Embedding-0.6B` |
| `CUDA_DEVICE` | GPU 设备编号 | `3` |
| `LLM_BASE_URL` | LLM API 地址 | `https://llm.3qiao.vip:23436/v1` |
| `LLM_API_KEY` | LLM API 密钥 | `sk-T3bQ...` |
| `LLM_MODEL` | LLM 模型名称 | `qwen2.5-72b-instruct-awq` |

---

## ✨ 核心功能详解

### 1. 跨页表格合并

**问题**：PDF 中的"分部分项工程量清单与计价表"经常跨多页，每页被识别为独立的 chunk，导致信息割裂。

**解决方案**：
- 在 chunking 时生成 `table_group_id`（基于表头 + stage）
- 使用 `TableMerger` 按 `table_group_id` 和 `page_idx` 自动合并连续页
- 合并后的 chunk 包含完整表格内容

**示例**：

```python
from finturn_embeding.services import merge_table_groups

# 原始：3个独立的 chunk（第5页、第6页、第7页）
chunks = [
    {"type": "table", "page_idx": 5, "table_group_id": "abc123", "text": "..."},
    {"type": "table", "page_idx": 6, "table_group_id": "abc123", "text": "..."},
    {"type": "table", "page_idx": 7, "table_group_id": "abc123", "text": "..."},
]

# 合并后：1个 chunk（第5-7页）
merged = merge_table_groups(chunks, max_gap=1)
# merged[0]["text"] = "【土石方工程】\n第5-7页表格（跨页合并）\n..."
# merged[0]["metadata"]["is_merged"] = True
# merged[0]["metadata"]["merged_pages"] = [5, 6, 7]
```

**配置**：
```python
config.enable_table_merge = True      # 启用合并
config.table_merge_max_gap = 1        # 最大页码间隔（1=仅连续页）
```

---

### 2. 纯 LLM Stage 分类

**问题**：工程阶段分类需要理解上下文，规则判断准确度低。

**解决方案**：
- 移除所有规则判断逻辑
- 使用 LLM 单选分类（7个工程阶段）
- 输入：chunk 文本（前1200字）+ 表头信息

**7个工程阶段**：
1. 土石方工程
2. 地基处理工程
3. 基坑支护工程
4. 主体工程
5. 装饰装修工程
6. 设备安装工程
7. 室外工程

**配置**：
```python
config.enable_stage_classification = True
config.stage_use_llm = True  # 使用 LLM（否则返回默认值"主体工程"）
```

---

### 3. 纯稠密向量（移除稀疏向量）

**原因**：qwen3-embedding-0.6b 不支持稀疏向量生成。

**改动**：
- 移除 `scipy.sparse` 依赖
- `milvus_adapter.insert_rows()` 不再插入 `sparse_vector` 字段
- 检索时仅使用 `dense_vector`

**注意**：如果 Milvus 集合 schema 中有 `sparse_vector` 字段，建议重建集合移除该字段。

---

## 📊 数据格式

### Chunks JSON 格式

```json
{
  "document_id": "100",
  "source_file": "100_origin.pdf",
  "chunks": [
    {
      "type": "table",                    // 或 "desc"
      "page_idx": 5,
      "chunk_index": 0,
      "construction_stage": "土石方工程",
      "table_group_id": "abc123xyz",      // ✨ 用于跨页合并
      "headers_norm": ["feature_desc", "unit", "quantity"],
      "table_markdown": "| 项目特征 | 单位 | 工程量 |\n|---|---|---|...",
      "text": "【土石方工程】\n第5页表格\n...",
      "content": "## 【土石方工程 - 第5页】\n\n...",
      "bbox": [x0, y0, x1, y1],
      "metadata": {
        "source": "pymupdf",
        "is_table": true,
        "is_merged": false,             // ✨ 是否合并后的 chunk
        "merged_pages": [],             // ✨ 合并的页码列表
        "table_rows": 25,
        "table_cols": 8,
        "text_length": 1234
      }
    }
  ],
  "markdown_tables": ["...", "..."],
  "total_chunks": 15,
  "total_tables": 8,
  "table_merged": true                  // ✨ 是否执行了表格合并
}
```

### 检索结果 CSV 格式

```csv
query,original_query,score,chunk_id,document_id,text,stage,page_idx,source_file,headers_norm,content,type
工程量,工程量,0.856,100-5-0,100,"【土石方工程】...",土石方工程,5,100_origin.pdf,"feature_desc|unit|quantity","## 【土石方工程 - 第5页】...",positive
```

---

## 🔍 故障排查

### 1. 模型加载失败

```bash
❌ Embedding 模型加载失败: [Errno 2] No such file or directory: '/path/to/model'
```

**解决**：检查 `EMBEDDING_MODEL_DIR` 环境变量或 `config.model_dir` 是否正确。

### 2. LLM API 调用失败

```bash
⚠️ LLM 分类失败: Connection timeout，使用默认值
```

**解决**：检查 `LLM_BASE_URL`、`LLM_API_KEY` 和网络连接。如果不需要 LLM 分类，可设置：
```python
config.stage_use_llm = False  # 使用默认值"主体工程"
```

### 3. Milvus 连接失败

```bash
❌ Milvus 连接失败: failed to connect to all addresses
```

**解决**：
1. 检查 Milvus 是否启动：`docker ps | grep milvus`
2. 检查端口：默认 `127.0.0.1:19530`

### 4. 表格未合并

**检查**：
```python
config.enable_table_merge = True  # 确保启用
```

查看 JSON 中的 `table_merged` 字段和 `metadata.is_merged` 字段。

---

## 🎯 业务目标

- **目标**：为 embedding 微调准备高质量训练数据
- **数据来源**：概算表 PDF 中的"分部分项工程量清单与计价表"
- **关键字段**：项目名称、项目特征描述、计量单位、工程量
- **输出**：
  1. Chunks JSON（分块 + 合并后的表格）
  2. 向量数据库（支持语义检索）
  3. 标注 CSV（positive/negative，用于微调）

---

## 🛠️ 开发与扩展

### 添加新的 Service

在 `services/` 下创建新文件，例如 `my_service.py`：

```python
"""我的自定义服务"""

class MyService:
    def process(self, input_data):
        # 实现你的逻辑
        return output_data

def my_function(input_data):
    """快捷函数"""
    service = MyService()
    return service.process(input_data)
```

在 `services/__init__.py` 中导出：

```python
from .my_service import MyService, my_function

__all__ = [..., 'MyService', 'my_function']
```

### 修改配置

编辑 `config.py` 中的 `PipelineConfig`：

```python
@dataclass
class PipelineConfig:
    # 添加新配置项
    my_custom_param: str = "default_value"
```

---

## 📝 更新日志

### v2.0.0（重构版）- 2025-01-10

**🎯 重大改进**：
- ✅ 实现跨页表格自动合并
- ✅ Stage 分类改为纯 LLM
- ✅ 移除稀疏向量（qwen3-0.6b 不支持）
- ✅ 创建 `services/` 原子能力层
- ✅ 统一配置类 `PipelineConfig`
- ✅ 删除所有冗余入口和兼容代码

**🗑️ 移除**：
- `step1_fetch_reports.py`
- `step2_ocr_chunking.py`
- `batch_pipeline.py`
- `ingest_from_reports.py`
- `retrieval_and_label_cli.py`
- `factor_retrieval_cli.py`
- 其他冗余文件

**🎯 唯一入口**：`cli.py`

---

## 📧 联系与支持

如有问题，请查看代码注释或联系开发团队。

**推荐工作流**：
```bash
# 1. 生成 chunks（自动合并表格）
python -m finturn_embeding.cli chunks --enable-ocr

# 2. 入库（LLM 分类 stage）
python -m finturn_embeding.cli ingest

# 3. 检索 + 打标
python -m finturn_embeding.cli retrieve --queries "工程量" "挖深"

# 4. 人工审阅标注结果
# 5. 用于 embedding 微调
```
