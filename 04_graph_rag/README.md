# GraphRAG 模块化架构

## 概述

本项目已重构为模块化架构，支持灵活的向量库扩展、文档导入、知识图谱构建和QA问答功能。

## 目录结构

```
04_graph_rag/
├── __init__.py                    # 包初始化
├── graph_rag.py                   # 主入口（导入新模块）
├── README.md                      # 本文件
│
├── core/                          # 核心模块
│   ├── __init__.py
│   ├── kg_service.py             # 知识图谱服务（组合各个模块）
│   └── config.py                 # 配置管理
│
├── vector_db/                     # 向量数据库抽象层
│   ├── __init__.py
│   ├── base.py                   # 向量库抽象接口
│   ├── milvus_adapter.py         # Milvus适配器
│   └── factory.py                # 向量库工厂
│
├── document_processor/            # 文档处理模块
│   ├── __init__.py
│   ├── base.py                   # 文档处理器接口
│   ├── file_importer.py          # 文件导入器（PDF, DOCX, TXT等）
│   └── chunker.py                # 文档分块器
│
├── knowledge_extractor/           # 知识抽取模块
│   ├── __init__.py
│   ├── entity_extractor.py       # 实体抽取器
│   ├── relation_extractor.py      # 关系抽取器
│   └── knowledge_extractor.py    # 统一的知识抽取器
│
├── storage/                       # 存储模块
│   ├── __init__.py
│   ├── graph_storage.py          # 图谱存储（NetworkX）
│   ├── db_storage.py             # 关系数据库存储
│   └── vector_storage.py         # 向量存储
│
└── query/                         # 查询模块
    ├── __init__.py
    ├── keyword_extractor.py      # 关键词提取器
    └── graphrag_query.py         # GraphRAG查询引擎
```

## 快速开始

### 基本使用

```python
from graph_rag import KGService, GraphRAGConfig

# 使用默认配置（Milvus）
kg_service = KGService()

# 处理文档
chunks = [
    {"text": "文档内容...", "chunk_id": "chunk1", "tokens": 100}
]
result = await kg_service.process_document_chunks(
    document_id=1,
    chunks=chunks,
    max_concurrent=5
)

# GraphRAG查询
query_result = await kg_service.graphrag_query(
    query="你的问题",
    mode="mix",  # "local", "global", "hybrid", "mix"
    top_k=10
)
```

### 使用不同向量库

```python
from graph_rag import KGService, GraphRAGConfig
from graph_rag.vector_db.factory import VectorDBFactory

# 创建配置
config = GraphRAGConfig(
    vector_db_type="milvus",  # 或 "chroma", "faiss"（未来支持）
    vector_db_config={"host": "localhost", "port": 19530}
)

# 创建向量库适配器
vector_db = VectorDBFactory.create("milvus", config.vector_db_config)

# 创建服务
kg_service = KGService(config=config, vector_db=vector_db)
```

### 文件导入

```python
from graph_rag.document_processor import FileImporter

importer = FileImporter()
chunks = importer.import_file(
    file_path="document.pdf",
    chunk_size=1000,
    chunk_overlap=200
)
```

## 核心模块说明

### 1. 向量数据库抽象层 (`vector_db/`)

- **VectorDBInterface**: 抽象接口，定义标准方法
- **MilvusAdapter**: Milvus实现
- **VectorDBFactory**: 工厂类，支持动态创建不同向量库适配器

**扩展新的向量库**：

```python
from graph_rag.vector_db.base import VectorDBInterface

class ChromaAdapter(VectorDBInterface):
    def connect(self):
        # 实现连接逻辑
        pass
    
    def search(self, ...):
        # 实现搜索逻辑
        pass
    # ... 实现其他方法

# 注册适配器
from graph_rag.vector_db.factory import VectorDBFactory
VectorDBFactory.register_adapter("chroma", ChromaAdapter)
```

### 2. 文档处理 (`document_processor/`)

- **FileImporter**: 支持PDF、DOCX、TXT等格式的文件导入
- **Chunker**: 支持多种分块策略（固定大小、按句子、按段落）

### 3. 知识抽取 (`knowledge_extractor/`)

- **EntityExtractor**: 从文本中抽取实体
- **RelationExtractor**: 从文本中抽取关系
- **KnowledgeExtractor**: 统一的知识抽取器（同时抽取实体和关系）

### 4. 存储 (`storage/`)

- **GraphStorage**: NetworkX图存储
- **DBStorage**: 关系数据库存储（实体、关系、元数据）
- **VectorStorage**: 向量存储（使用向量数据库抽象层）

### 5. 查询 (`query/`)

- **KeywordExtractor**: 从查询中提取高层和低层关键词
- **GraphRAGQueryEngine**: GraphRAG查询引擎
  - `local_search`: 基于实体的检索
  - `global_search`: 基于关系的检索
  - `vector_search`: 直接向量检索
  - `graphrag_query`: 统一查询接口

## 设计模式

### 依赖注入

所有模块都通过依赖注入的方式组合，便于测试和扩展：

```python
kg_service = KGService(
    vector_db=custom_vector_db,
    document_processor=custom_processor,
    knowledge_extractor=custom_extractor,
    # ...
)
```

### 工厂模式

使用工厂模式创建向量库适配器：

```python
vector_db = VectorDBFactory.create("milvus", config)
```

## 优势

1. **易于扩展**：添加新向量库只需实现 `VectorDBInterface` 接口
2. **职责清晰**：每个模块专注单一职责
3. **易于测试**：可以单独测试每个模块
4. **代码复用**：模块可以在其他项目中复用
5. **维护性好**：修改一个模块不影响其他模块

## 迁移指南

### 从旧版本迁移

旧版本代码：
```python
from graph_rag import KGService

kg_service = KGService(model_type=LlmClientType.DeepSeekV3)
```

新版本代码（兼容）：
```python
from graph_rag import KGService

kg_service = KGService(model_type=LlmClientType.DeepSeekV3)
```

新版本代码（推荐）：
```python
from graph_rag import KGService, GraphRAGConfig

config = GraphRAGConfig(model_type=LlmClientType.DeepSeekV3)
kg_service = KGService(config=config)
```

## 未来扩展

- [ ] 支持更多向量库（ChromaDB, FAISS等）
- [ ] 支持更多文档格式
- [ ] 支持更多分块策略
- [ ] 支持批量文档处理
- [ ] 支持增量更新

