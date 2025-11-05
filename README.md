# RAG的道与术 🚀

> 从零开始学习检索增强生成（Retrieval-Augmented Generation）技术，掌握企业级知识库搭建的完整方法论

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green.svg)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

在AI大模型时代，我们经常需要让AI处理企业内部文档、专有数据等非公开信息。但由于这些内容不在模型的训练范围内，就会出现知识盲区。**RAG技术**很好地解决了这个问题 - 它能让AI模型实时获取并理解我们提供的专有知识，从而针对特定领域或场景提供更加准确和相关的回答。

本项目是一个**系统化的RAG实战教程**，涵盖了从基础概念到高级应用的完整知识体系，特别关注实际项目中遇到的各种问题及其解决方案。

### ✨ 适合人群

- 🎓 正在学习大模型应用开发的同学
- 💼 需要搭建企业知识库的工程师
- 🔧 想要解决RAG实践问题的开发者
- 🧪 对AI技术感兴趣的研究人员

## 🌟 为什么选择本教程

- **从问题出发**：不只是介绍技术，更关注实际项目中的"七大坑"及解决方案
- **代码可运行**：每个概念都配有可执行的Jupyter Notebook代码示例
- **架构清晰**：按照"输入优-查得准-表达好"的优化思路组织内容
- **持续更新**：跟踪最新技术趋势，不断补充新内容

## 📚 目录结构

```
RAG/
├── README.md                          # 项目总览（本文件）
├── requirements.txt                   # Python依赖包
├── .gitignore                         # Git忽略配置
│
├── 📘 docs/                           # 基础理论文档
│   └── Readme.md                      # RAG基础概念与背景知识
│
├── 🔧 01_data_clean/                  # 第一步：数据清洗与预处理
│   └── data_process.ipynb             # 数据格式化、图片表格处理
│
├── 🔍 02_data_retriever/              # 第二步：检索技术
│   ├── data_retriever.ipynb           # 向量、关键词、图检索对比
│   ├── embedding_models.md            # Embedding模型选择指南
│   ├── embedding_compare.ipynb        # 模型效果对比实验（待开发）
│   ├── embedding_finetune.ipynb       # 模型微调教程（待开发）
│   └── rag_teach/                     # 示例知识库
│       ├── rag_teach.faiss            # FAISS向量索引
│       └── rag_teach.pkl              # 向量数据
│
├── 🎯 03_llm_answer/                  # 第三步：LLM答案生成
│   ├── llm_answer.ipynb               # 基础RAG：Prompt优化、答案生成
│   └── media_coverage_report.docx     # 示例文档
│
├── 🕸️ 04_graph_rag/                   # 第四步：图检索（GraphRAG）
│   ├── README.md                      # GraphRAG原理与应用场景
│   ├── README_arch.md                 # 模块化架构说明
│   ├── graph_rag_basic.ipynb          # 基础图检索：Neo4j + LangChain（待开发）
│   └── graphrag_microsoft.ipynb       # Microsoft GraphRAG实现（待开发）
│
├── 🤖 05_rag_agent/                   # 第五步：RAG智能体（高级）
│   ├── base_rag_agent.ipynb           # 智能RAG：回答风格、记忆管理
│   └── extend_adaptive_rag.ipynb      # 自适应RAG：路由、联网补充
│
├── 📊 06_rag_evaluation/              # 第六步：效果评估（开发中）
│   └── (待补充评估方法)
│
├── 🗄️ 07_vector_database_enterprise/  # 第七步：企业级向量数据库
│   ├── README.md                      # 向量数据库选型与架构
│   ├── milvus_architecture.md         # Milvus架构深度解析
│   ├── interview_guide.md             # 面试答题模板
│   ├── milvus_basic_setup.ipynb       # Milvus基础部署与使用（待开发）
│   ├── milvus_segment_management.ipynb # Segment生命周期管理（待开发）
│   ├── milvus_news_system.ipynb       # 企业新闻检索系统完整实现（待开发）
│   └── milvus_production_config.yaml  # 生产环境配置模板
│
├── 🎨 08_multimodal_embedding/        # 第八步：多模态嵌入（新增）
│   ├── README.md                      # 多模态嵌入原理与应用
│   ├── clip_basics.ipynb              # CLIP图文检索入门（待开发）
│   ├── ecommerce_image_search.ipynb   # 电商产品搜索系统（待开发）
│   ├── photo_album_search.ipynb       # 智能相册管理（待开发）
│   ├── video_search.ipynb             # 视频内容检索（待开发）
│   └── multimodal_rag_system.ipynb    # 多模态RAG系统（待开发）
│
├── 🚀 future_practice/                # 技术探索与未来方向
│   └── Readme.md                      # 待实践的技术清单
│
├── 📄 paper_learning/                 # 相关论文学习
│   └── GraphRAG综述.pdf
│
├── 📎 示例文档/
│   ├── LLaMA2.pdf                     # 测试用PDF文档
│   └── LLaVA.pdf                      # 测试用PDF文档
│
├── 🖼️ images/ & imgs/                 # 文档图片资源
│
├── main.py                            # FastAPI服务主程序
└── image_rag.ipynb                    # 多模态RAG示例

```

## 🎯 学习路线

### 第一阶段：理解RAG核心概念

**目标**：掌握RAG的基本原理和应用场景

1. 📖 阅读 [基础知识文档](./docs/Readme.md)
   - RAG是什么？为什么需要RAG？
   - RAG的核心步骤
   - RAG的七大常见问题

**学习时间**：30分钟

### 第二阶段：输入优化 - 让数据更干净

**目标**：解决图片、表格等复杂格式的数据处理问题

2. 🔧 实践 [数据清洗](./01_data_clean/data_process.ipynb)
   - 内容精简与格式标准化
   - 图片内容提取（多模态）
   - 表格转文本方法

3. 🚀 进阶 [自适应RAG](./04_rag_agent/extend_adaptive_rag.ipynb)
   - 智能文档解析
   - 数据质量控制

**解决问题**：
- ✅ 图片内容无法导入
- ✅ 表格内容检索差

**学习时间**：2-3小时

### 第三阶段：查得准 - 选择最佳检索方式

**目标**：根据场景选择合适的检索技术

4. 🔍 实践 [检索技术对比](./02_data_retriever/data_retriever.ipynb)
   - 向量检索（Embedding + Rerank）
   - 关键词检索（BM25）
   - 图检索（GraphRAG）
   - 混合检索策略

5. 📚 进阶 [Embedding模型选择](./02_data_retriever/embedding_models.md)
   - OpenAI vs BGE vs JinaAI
   - 中文模型推荐
   - 成本与性能平衡
   - 模型选择决策树

6. 🕸️ 专题 [GraphRAG图检索](./04_graph_rag/README.md)
   - 知识图谱构建
   - Neo4j + LangChain
   - GraphRAG vs 向量检索
   - 适用场景判断

**解决问题**：
- ✅ 提问总是检索不到对应的文本块
- ✅ 对长文本提问总结性问题总是回答不全
- ✅ 复杂关系查询效果差

**学习时间**：6-8小时

### 第四阶段：表达好 - 优化回答质量

**目标**：让AI回答更准确、更符合需求

7. 🎯 实践 [LLM答案生成](./03_llm_answer/llm_answer.ipynb)
   - Prompt工程基础
   - 基于检索内容生成答案
   - 答案结构化
   - 引用来源标注

**解决问题**：
- ✅ 查到的知识是对的，但是回答的语气、口吻和精细度不佳

**学习时间**：2-3小时

### 第五阶段：高级应用 - RAG智能体

**目标**：实现更智能的RAG系统

8. 🤖 实践 [基础RAG智能体](./05_rag_agent/base_rag_agent.ipynb)
   - 回答风格和语气调整
   - 对话记忆管理
   - 避免编造和幻觉

9. 🚀 进阶 [自适应RAG](./05_rag_agent/extend_adaptive_rag.ipynb)
   - 智能路由：知识库查询 vs 联网查询
   - 知识补充：缺失知识的联网补充
   - 准确性控制：强制基于知识库回答

**解决问题**：
- ✅ 知识库特别大，难以完全测试来保证准确度

**学习时间**：3-4小时

### 第六阶段：企业级实战 - 向量数据库深度

**目标**：掌握生产级向量数据库部署和优化

10. 🗄️ 深入 [Milvus架构原理](./07_vector_database_enterprise/milvus_architecture.md)
   - 7层数据组织架构
   - Growing/Sealed Segment机制
   - Flush与Compaction策略
   - 三层存储架构设计

11. 💼 面试 [向量数据库面试指南](./07_vector_database_enterprise/interview_guide.md)
   - 高频面试问题
   - 标准答题模板
   - 实战场景设计

**解决问题**：
- ✅ FAISS无法满足大规模生产需求
- ✅ 不了解向量数据库底层原理
- ✅ 面试中无法深入讲解技术细节

**学习时间**：4-6小时

### 第七阶段：多模态扩展 - 突破文本局限

**目标**：学习多模态嵌入技术，实现图文音视频联合检索

12. 🎨 探索 [多模态嵌入](./08_multimodal_embedding/README.md)
   - CLIP图文联合嵌入
   - 以图搜图、以文搜图
   - 视频内容检索
   - 多模态RAG系统

**解决问题**：
- ✅ 纯文本RAG无法处理图像内容
- ✅ 需要实现以图搜图功能
- ✅ 视频、音频检索需求
- ✅ 构建更丰富的多模态应用

**学习时间**：3-5小时（理论）+ 按需实践

## 🔥 快速开始

### 环境要求

- Python 3.10+
- 8GB+ RAM
- OpenAI API 或 Azure OpenAI 服务

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd RAG

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
# 创建 .env 文件，添加以下内容：
# AZURE_OPENAI_API_KEY=your_api_key
# AZURE_OPENAI_ENDPOINT=your_endpoint
# 或使用 OpenAI：
# OPENAI_API_KEY=your_api_key

# 5. 启动Jupyter
jupyter notebook
```

### 运行第一个示例

打开 `02_data_retriever/data_retriever.ipynb`，按顺序运行代码单元，体验向量检索的完整流程。

### 运行Web服务

```bash
# 启动FastAPI服务
python main.py

# 访问 http://localhost:8005
# API文档：http://localhost:8005/docs
```

## 📊 RAG优化三原则

### 1. 输入优 - 避免添加无效内容

- **内容精简**：只导入会被提问到的资料
- **内容准确**：避免歧义和矛盾的描述
- **格式标准化**：图片、表格转为可检索的文本

**推荐工具**：[OmniParse](https://github.com/adithya-s-k/omniparse)

### 2. 查得准 - 选择合适的检索方式

| 检索方式 | 推荐场景 | 优点 | 缺点 |
|---------|---------|------|------|
| **向量检索** | 规章条文等上下文关联度不大的内容 | 速度快，能够语义检索 | 对数字关键词不敏感，可能遗漏信息 |
| **图检索** | 小说、人物关系等复杂上下文内容 | 包括全文相关内容，不受长度限制 | 构建知识图谱成本高 |
| **关键词检索** | 有具体数字和数值的内容 | 精准匹配 | 需要记住关键词 |

### 3. 表达好 - 优化回答风格

- **Prompt工程**：设计合适的系统提示词
- **风格控制**：调整回答的正式程度、详细程度
- **记忆管理**：保持上下文连贯性
- **幻觉控制**：避免AI编造不存在的信息

## 🛠️ 技术栈

- **LLM框架**：LangChain
- **大模型**：OpenAI GPT-4 / Azure OpenAI
- **向量数据库**：FAISS、ChromaDB
- **Embedding模型**：OpenAI text-embedding-3-small
- **文档处理**：Unstructured、pdf2image
- **Web框架**：FastAPI
- **开发环境**：Jupyter Notebook

## 🤝 参与贡献

我们欢迎大家一起参与到RAG技术的探索中来！

### 如何贡献

1. 🐛 **提交Issue**：发现bug或有改进建议
2. 💡 **提出想法**：在 [future_practice](./future_practice/Readme.md) 中添加你想探索的技术
3. 📝 **改进文档**：完善教程内容或修正错误
4. 💻 **贡献代码**：提交Pull Request

### 待实践的技术（欢迎认领）

已完成 ✅：
- [x] Embedding模型选择指南（已完成）
- [x] GraphRAG原理与应用（已完成）
- [x] Milvus企业级架构（已完成）
- [x] 向量数据库面试指南（已完成）

待开发 📋：
- [ ] Embedding模型对比实验（02_data_retriever/embedding_compare.ipynb）
- [ ] Embedding模型微调方法（02_data_retriever/embedding_finetune.ipynb）
- [ ] Neo4j图检索实战（04_graph_rag/graph_rag_basic.ipynb）
- [ ] Microsoft GraphRAG实现（04_graph_rag/graphrag_microsoft.ipynb）
- [ ] Milvus基础部署教程（07_vector_database_enterprise/milvus_basic_setup.ipynb）
- [ ] Segment管理实战（07_vector_database_enterprise/milvus_segment_management.ipynb）
- [ ] 企业新闻检索系统（07_vector_database_enterprise/milvus_news_system.ipynb）
- [ ] CLIP图文检索入门（08_multimodal_embedding/clip_basics.ipynb）
- [ ] 电商产品搜索系统（08_multimodal_embedding/ecommerce_image_search.ipynb）
- [ ] 智能相册管理（08_multimodal_embedding/photo_album_search.ipynb）
- [ ] 视频内容检索（08_multimodal_embedding/video_search.ipynb）
- [ ] 多模态RAG系统（08_multimodal_embedding/multimodal_rag_system.ipynb）
- [ ] RAG自动评估框架
- [ ] LightRAG实现与对比
- [ ] 更多期待你的贡献...

详见：[future_practice](./future_practice/Readme.md)

## 📖 相关资源

### 推荐阅读

- [LangChain官方文档](https://python.langchain.com/)
- [OpenAI Embeddings指南](https://platform.openai.com/docs/guides/embeddings)
- [RAG论文合集](./paper_learning/)

### 相关项目

- [OmniParse](https://github.com/adithya-s-k/omniparse) - 多格式文档解析工具
- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 📮 提交 Issue
- 💬 参与 Discussions
- 📧 发送邮件

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

**Happy Learning! 🎉**

