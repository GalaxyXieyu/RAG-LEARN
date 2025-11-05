# 第二步：数据检索技术 🔍

## 模块概述

检索是RAG系统的核心环节。本模块介绍不同的检索技术及其适用场景，帮助你根据实际需求选择最合适的检索方案。

## 学习目标

- ✅ 理解向量检索、关键词检索、图检索的原理
- ✅ 掌握Embedding模型和Rerank模型的使用
- ✅ 学会根据场景选择合适的检索方式
- ✅ 了解混合检索策略

## 三大检索方式对比

### 1. 向量检索（Semantic Search）

**原理**：将文本转换为向量（Embedding），通过向量相似度检索语义相关的内容。

**适用场景**：
- 规章条文、政策文档
- FAQ问答库
- 上下文关联度不大的独立内容

**优点**：
- ✅ 速度快（毫秒级）
- ✅ 支持语义理解（"手机"能匹配到"智能设备"）
- ✅ 技术成熟，生态丰富

**缺点**：
- ❌ 对数字、日期等关键词不敏感
- ❌ 基于文本切块，可能遗漏上下文信息
- ❌ 需要额外的Embedding模型

**技术栈**：
```python
# Embedding模型
- OpenAI text-embedding-3-small/large
- BGE系列（中文效果好）
- JinaAI Embedding

# 向量数据库
- FAISS（轻量级，适合小规模）
- ChromaDB（开发友好）
- Milvus（适合大规模）

# Rerank模型（提升准确率）
- Cohere Rerank
- BGE Reranker
```

### 2. 关键词检索（Keyword Search）

**原理**：基于BM25算法，通过关键词匹配检索相关文档。

**适用场景**：
- 包含具体数字、代码、专有名词的内容
- 需要精确匹配的场景
- 法律条文、技术文档

**优点**：
- ✅ 精准匹配关键词
- ✅ 对数字、符号敏感
- ✅ 无需额外模型，成本低

**缺点**：
- ❌ 不理解语义（"手机"不能匹配"智能设备"）
- ❌ 需要用户准确输入关键词
- ❌ 同义词识别能力弱

**技术栈**：
```python
# BM25算法
from rank_bm25 import BM25Okapi

# Elasticsearch
- 全文检索引擎
- 支持中文分词
- 企业级特性
```

### 3. 图检索（Graph RAG）

**原理**：构建知识图谱，通过实体和关系检索相关信息。

**📘 完整教程**：[GraphRAG 图检索专题](../04_graph_rag/README.md)

**适用场景**：
- 小说、故事（人物关系复杂）
- 企业组织架构
- 需要推理和多跳查询的场景
- 长文档总结

**优点**：
- ✅ 保留完整上下文关系
- ✅ 支持复杂推理查询
- ✅ 不受文档长度限制
- ✅ 能发现隐藏的关联

**缺点**：
- ❌ 构建成本高（需要调用LLM提取关系）
- ❌ 实时性较差
- ❌ 技术复杂度高

**技术栈**：
```python
# GraphRAG实现
- Microsoft GraphRAG
- LightRAG（轻量级）
- Neo4j + LangChain
```

**深入学习**：
- 📓 [Neo4j图检索实战](../04_graph_rag/graph_rag_basic.ipynb)（待开发）
- 📓 [Microsoft GraphRAG](../04_graph_rag/graphrag_microsoft.ipynb)（待开发）

## 检索方式选择指南

| 检索方式 | 推荐场景 | 典型应用 |
|---------|---------|---------|
| **向量检索** | 独立、短文本知识库 | 客服FAQ、政策问答 |
| **关键词检索** | 精确匹配需求 | 法律条文查询、代码搜索 |
| **图检索** | 复杂关系、长文档 | 小说分析、企业知识图谱 |
| **混合检索** | 综合性知识库 | 企业文档库、技术支持 |

## 混合检索策略

实际项目中，往往需要结合多种检索方式：

```python
# 策略1：并行检索 + Rerank
vector_results = vector_search(query, top_k=20)
keyword_results = keyword_search(query, top_k=20)
merged_results = merge_and_rerank(vector_results + keyword_results)

# 策略2：分类路由
if is_exact_match_query(query):
    results = keyword_search(query)
elif is_relationship_query(query):
    results = graph_search(query)
else:
    results = vector_search(query)

# 策略3：瀑布式检索
results = vector_search(query, threshold=0.8)
if len(results) < 3:
    results += keyword_search(query)
```

## 向量检索核心技术

### Embedding模型选择

**📘 完整指南**：[Embedding模型选择完全指南](./embedding_models.md)

**快速推荐**：

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 中文为主 | bge-large-zh-v1.5 | 中文效果最好，免费 |
| 多语言 | text-embedding-3-large | 综合效果最好 |
| 成本敏感 | bge-large-zh-v1.5 | 开源免费，自部署 |
| 性价比 | text-embedding-3-small | 效果好，成本低 |
| 长文本 | JinaAI embeddings-v3 | 支持 8K tokens上下文 |

**评测参考**：[MTEB排行榜](https://huggingface.co/spaces/mteb/leaderboard)

**进阶学习**：
- 📓 [Embedding模型对比实验](./embedding_compare.ipynb)（待开发）
- 📓 [Embedding模型微调教程](./embedding_finetune.ipynb)（待开发）

### Rerank模型使用

Rerank在检索结果中进行二次排序，显著提升准确率：

```python
# 步骤1：粗召回（向量检索，top 50）
candidates = vector_search(query, top_k=50)

# 步骤2：精排序（Rerank，top 5）
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank(model="rerank-multilingual-v2.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

final_results = compression_retriever.get_relevant_documents(query)
```

### 向量数据库选择

| 数据库 | 适用规模 | 特点 |
|--------|---------|------|
| **FAISS** | < 100万条 | 轻量级，本地部署 |
| **ChromaDB** | < 1000万条 | 开发友好，内置Embedding |
| **Milvus** | > 1000万条 | 企业级，分布式 |
| **Qdrant** | 任意规模 | 高性能，Rust编写 |

## 实践练习

### 📓 Notebook：data_retriever.ipynb

本模块的Jupyter Notebook包含：

1. **向量检索实战**
   - FAISS索引构建
   - Embedding模型使用
   - Rerank优化

2. **关键词检索实战**
   - BM25算法实现
   - 中文分词处理

3. **混合检索实战**
   - 结果合并策略
   - 相关性评分

4. **效果对比**
   - 不同检索方式的准确率对比
   - 响应时间对比

### 运行步骤

```bash
# 1. 启动Jupyter
jupyter notebook data_retriever.ipynb

# 2. 准备测试数据（已包含示例）
# - rag_teach.faiss：向量索引
# - rag_teach.pkl：文档数据

# 3. 运行对比实验
```

## 参数调优

### 向量检索参数

```python
# 1. top_k：返回结果数量
top_k = 5  # 平衡准确率和延迟

# 2. 相似度阈值
similarity_threshold = 0.7  # 过滤低相关结果

# 3. chunk_size：文本分块大小
chunk_size = 500  # 根据内容类型调整
chunk_overlap = 50  # 保留上下文
```

### BM25参数

```python
# 1. k1：词频饱和度
k1 = 1.5  # 默认值，一般不需要调整

# 2. b：文档长度惩罚
b = 0.75  # 默认值

# 3. 分词器
import jieba
tokenizer = jieba.cut  # 中文分词
```

## 常见问题

### Q1: 向量检索效果不好怎么办？
**A**: 按顺序尝试：
1. 检查Embedding模型是否适合中文（如使用BGE）
2. 调整chunk_size和overlap
3. 添加Rerank模型
4. 考虑混合检索

### Q2: 如何判断应该用哪种检索方式？
**A**: 
- 做一些测试查询，看哪种方式效果好
- 分析知识库特点：独立内容→向量，复杂关系→图
- 考虑用户查询习惯：精确关键词→BM25

### Q3: Rerank模型是否必须？
**A**: 
- 数据量小（< 1000条）：可选
- 数据量大或准确率要求高：建议使用
- 成本敏感：用开源BGE Reranker

## 下一步

完成检索技术学习后，继续：

➡️ [第三步：LLM答案生成](../03_llm_answer/README.md)  
➡️ [第四步：GraphRAG图检索](../04_graph_rag/README.md)  
➡️ [第七步：企业级向量数据库](../07_vector_database_enterprise/README.md)

---

💡 **提示**："查得准"是RAG系统的关键！建议针对自己的数据做A/B测试，找到最佳检索策略。

