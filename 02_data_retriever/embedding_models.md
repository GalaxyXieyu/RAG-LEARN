# Embedding 模型选择完全指南 🎯

> 选对模型，检索效果提升50%！

## 📖 目录

- [一、Embedding 模型基础](#一embedding-模型基础)
- [二、主流模型对比](#二主流模型对比)
- [三、模型选择决策树](#三模型选择决策树)
- [四、详细模型介绍](#四详细模型介绍)
- [五、成本与性能平衡](#五成本与性能平衡)
- [六、模型评测方法](#六模型评测方法)
- [七、实战建议](#七实战建议)

---

## 一、Embedding 模型基础

### 1.1 什么是 Embedding？

**Embedding**（嵌入/向量化）是将文本转换为高维向量的过程：

```python
text = "今天天气很好"
embedding = model.encode(text)
# 输出：[0.23, -0.45, 0.67, ..., 0.12]  # 768维或1024维向量
```

**作用**：将语义相似的文本映射到向量空间中距离较近的位置。

```
"天气不错" 的向量    → [0.22, -0.43, 0.65, ...]
"今天天气很好" 的向量 → [0.23, -0.45, 0.67, ...]  ← 距离很近！
"股票大跌" 的向量    → [-0.80, 0.30, -0.15, ...] ← 距离很远
```

### 1.2 Embedding 在 RAG 中的角色

```
RAG检索流程：
1. 离线：文档 → Embedding模型 → 向量 → 向量数据库
2. 在线：用户问题 → Embedding模型 → 向量 → 检索相似向量 → 返回文档
```

**关键点**：
- ⚠️ 离线和在线必须使用**同一个**Embedding模型
- ⚠️ 模型维度一旦确定，后续更换成本极高（需要重建索引）
- ⚠️ 模型选择会直接影响检索的**召回率**和**准确率**

---

## 二、主流模型对比

### 2.1 综合对比表

| 模型 | 维度 | 语言 | MTEB得分 | 速度 | 成本 | 推荐度 |
|------|------|------|---------|------|------|--------|
| **OpenAI text-embedding-3-large** | 3072 | 多语言 | 64.6 | 快 | $$$ | ⭐⭐⭐⭐⭐ |
| **OpenAI text-embedding-3-small** | 1536 | 多语言 | 62.3 | 快 | $$ | ⭐⭐⭐⭐⭐ |
| **OpenAI text-embedding-ada-002** | 1536 | 多语言 | 60.9 | 快 | $$ | ⭐⭐⭐⭐ |
| **bge-large-zh-v1.5** | 1024 | 中文优化 | 64.5(中) | 中 | 免费 | ⭐⭐⭐⭐⭐ |
| **bge-m3** | 1024 | 多语言 | 64.0 | 中 | 免费 | ⭐⭐⭐⭐ |
| **JinaAI embeddings-v2** | 768 | 多语言 | 60.4 | 快 | $$$ | ⭐⭐⭐⭐ |
| **JinaAI embeddings-v3** | 1024 | 多语言 | 65.5 | 快 | $$$ | ⭐⭐⭐⭐⭐ |
| **m3e-large** | 1024 | 中文 | 63.5(中) | 中 | 免费 | ⭐⭐⭐ |
| **gte-large-zh** | 1024 | 中文 | 62.8(中) | 中 | 免费 | ⭐⭐⭐ |

**图例**：
- **$$$$**: 非常贵 (> $0.0001/1K tokens)
- **$$$**: 贵 ($0.00005-$0.0001/1K tokens)
- **$$**: 中等 ($0.00001-$0.00005/1K tokens)
- **$**: 便宜 (< $0.00001/1K tokens)
- **免费**: 开源模型，自部署

### 2.2 场景推荐速查表

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **中文为主** | bge-large-zh-v1.5 | 中文效果最好，免费 |
| **多语言** | text-embedding-3-large | 综合效果最好 |
| **成本敏感** | bge-large-zh-v1.5 / bge-m3 | 开源免费，自部署 |
| **性价比** | text-embedding-3-small | 效果好，成本低 |
| **快速原型** | text-embedding-3-small | API调用，无需部署 |
| **企业生产** | bge-large-zh-v1.5 (自部署) | 成本可控，效果稳定 |
| **长文本（> 8K）** | JinaAI embeddings-v3 | 支持 8K tokens上下文 |

---

## 三、模型选择决策树

```
开始
  │
  ├─ 预算充足？
  │   ├─ 是 → 数据主要是中文？
  │   │        ├─ 是 → text-embedding-3-large 或 JinaAI v3
  │   │        └─ 否 → text-embedding-3-large
  │   │
  │   └─ 否 → 能自己部署吗？
  │            ├─ 是 → 主要中文？
  │            │      ├─ 是 → bge-large-zh-v1.5 (推荐)
  │            │      └─ 否 → bge-m3
  │            │
  │            └─ 否 → text-embedding-3-small (性价比之选)
  │
  └─ 特殊需求？
       ├─ 长文本（> 512 tokens） → JinaAI v3 (8K上下文)
       ├─ 多语言混合 → bge-m3 或 text-embedding-3-large
       ├─ 领域特化 → 微调 bge-large-zh-v1.5
       └─ 极致性能 → text-embedding-3-large
```

---

## 四、详细模型介绍

### 4.1 OpenAI Embedding 系列

#### text-embedding-3-large

**特点**：
- ✅ 综合效果最好（MTEB 64.6）
- ✅ 支持多语言（100+ 语言）
- ✅ 维度高（3072），捕获更多语义
- ✅ API 调用，无需部署

**适用场景**：
```
1. 对检索精度要求极高
2. 多语言混合文档
3. 快速上线，无运维压力
4. 预算充足（$0.00013/1K tokens）
```

**代码示例**：
```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-large",
    input="今天天气很好",
    encoding_format="float"
)

embedding = response.data[0].embedding  # 3072维向量
```

**成本计算**：
```
100万条文档，平均500 tokens/条
= 500M tokens
= 500,000 * $0.00013
= $65

月查询量 100万次，平均20 tokens/次
= 20M tokens  
= 20,000 * $0.00013
= $2.6

总成本：$67.6/月（一次性建库 + 月查询）
```

#### text-embedding-3-small

**特点**：
- ✅ 性价比最高（效果好 + 价格低）
- ✅ 1536维，平衡性能和成本
- ✅ MTEB 62.3（仅比 large 低 2.3 分）
- ✅ 价格仅为 large 的 1/5

**适用场景**：
```
1. 中小型项目（< 1000万文档）
2. 对成本敏感
3. 不追求极致精度
4. 快速验证想法
```

**代码示例**：
```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="今天天气很好"
)

embedding = response.data[0].embedding  # 1536维向量
```

**成本计算**：
```
同样场景（100万文档，月查询100万次）
= 500,000 * $0.00002 + 20,000 * $0.00002
= $10 + $0.4
= $10.4/月（仅为 large 的 15%！）
```

**推荐理由**：
> 对于大多数应用，small 版本的效果已经足够好，性价比极高！

---

### 4.2 BGE（BAAI General Embedding）系列

#### bge-large-zh-v1.5

**特点**：
- ✅ 中文效果最好（C-MTEB 64.5）
- ✅ 完全免费，可自部署
- ✅ 1024维，内存占用适中
- ✅ 活跃维护，社区支持好

**适用场景**：
```
1. 中文为主的应用（90%+ 中文）
2. 需要自己掌控数据
3. 长期项目，降低成本
4. 企业内部部署
```

**代码示例**：
```python
from sentence_transformers import SentenceTransformer

# 本地加载模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 编码
embedding = model.encode("今天天气很好")  # 1024维向量

# 批量编码（更高效）
texts = ["文本1", "文本2", "文本3"]
embeddings = model.encode(texts, batch_size=32)
```

**部署成本**：
```
硬件需求：
├─ GPU：NVIDIA T4 或更好（推荐 A10/A100）
├─ 内存：16GB+
├─ 存储：10GB（模型 + 依赖）

云服务器（按量计费）：
├─ AWS g4dn.xlarge (T4): $0.526/小时 = $379/月
├─ 阿里云 T4: ¥2.5/小时 = ¥1,800/月

性能：
├─ T4 GPU: ~500 sentences/秒
├─ A10 GPU: ~2000 sentences/秒
└─ 100万文档向量化：30-120分钟
```

**与 OpenAI 成本对比**：
```
场景：100万文档，月查询100万次

OpenAI small: $10.4/月（持续付费）
BGE自部署: ¥1,800/月（服务器）+ 一次性向量化

Break-even point: 
├─ 如果项目运行 > 6个月 → BGE 更省钱
├─ 如果项目 < 6个月 → OpenAI 更省钱
```

#### bge-m3

**特点**：
- ✅ 支持 100+ 语言
- ✅ 多功能：稀疏检索 + 密集检索 + 多向量
- ✅ 免费开源
- ✅ 统一模型处理多语言

**适用场景**：
```
1. 多语言混合文档
2. 需要稀疏+密集混合检索
3. 希望一个模型搞定所有语言
```

**代码示例**：
```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 密集向量
embeddings = model.encode(
    ["文本1", "text 2", "テキスト3"],  # 中英日混合
    batch_size=12,
    max_length=8192
)['dense_vecs']

# 稀疏向量（用于混合检索）
sparse_embeddings = model.encode(
    texts,
    return_sparse=True
)['lexical_weights']
```

---

### 4.3 JinaAI Embedding 系列

#### JinaAI embeddings-v3

**特点**：
- ✅ 支持 **8K tokens** 上下文（最长！）
- ✅ 多任务支持：检索、分类、聚类
- ✅ MTEB 65.5（非常高）
- ✅ API 调用，简单易用

**适用场景**：
```
1. 长文档嵌入（> 512 tokens）
2. 需要完整段落/章节的语义
3. 不想处理文本切块
4. 对精度要求高
```

**长文本优势**：
```
传统模型（512 tokens）：
├─ 长文档 → 强制切块 → 上下文丢失
└─ 示例："第一章...第二章..." → 只能分开处理

JinaAI v3（8K tokens）：
├─ 长文档 → 整体嵌入 → 上下文完整
└─ 示例："第一章...第二章..." → 完整处理
```

**代码示例**：
```python
import requests

url = 'https://api.jina.ai/v1/embeddings'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
}

data = {
    "model": "jina-embeddings-v3",
    "task": "retrieval.passage",  # 或 retrieval.query
    "input": ["很长的文档内容..." * 100]  # 支持 8K tokens
}

response = requests.post(url, headers=headers, json=data)
embedding = response.json()['data'][0]['embedding']
```

**成本**：
```
价格：$0.00002/1K tokens（与 OpenAI small 相同）

适用场景：
├─ 长文档多 → JinaAI v3（无需切块）
├─ 短文档多 → OpenAI small（更成熟）
```

---

### 4.4 其他开源模型

#### m3e-large（中文）

**特点**：
- 专注中文
- 基于中文数据训练
- 免费开源

**劣势**：
- 社区支持不如 BGE
- 更新频率较低

**推荐度**：⭐⭐⭐（不如 bge-large-zh-v1.5）

#### gte-large-zh（中文）

**特点**：
- 阿里达摩院开源
- 中文效果好

**劣势**：
- 生态不如 BGE
- 文档相对较少

**推荐度**：⭐⭐⭐（可作为 BGE 的备选）

---

## 五、成本与性能平衡

### 5.1 成本对比（100万文档 + 月查询100万次）

| 方案 | 建库成本 | 月查询成本 | 总成本/月 | 性能 |
|------|---------|-----------|----------|------|
| **OpenAI large** | $65 | $2.6 | $67.6 | 最好 |
| **OpenAI small** | $10 | $0.4 | $10.4 | 很好 |
| **JinaAI v3** | $10 | $0.4 | $10.4 | 很好 |
| **BGE (自部署)** | ¥100(电费) | - | ¥1,800(服务器) | 很好 |

### 5.2 推荐组合

#### 方案1：小项目/快速验证
```
推荐：OpenAI text-embedding-3-small
理由：
├─ 快速上线（无需部署）
├─ 成本可控（$10/月）
├─ 效果足够好
└─ 后续可平滑迁移
```

#### 方案2：中文企业项目
```
推荐：bge-large-zh-v1.5 (自部署)
理由：
├─ 中文效果最好
├─ 长期成本低（6个月后回本）
├─ 数据可控（内网部署）
└─ 可定制微调
```

#### 方案3：多语言项目
```
推荐：text-embedding-3-large
理由：
├─ 100+ 语言支持
├─ 效果最稳定
├─ 无需维护多个模型
└─ API 调用省心
```

#### 方案4：长文档项目
```
推荐：JinaAI embeddings-v3
理由：
├─ 8K tokens 上下文
├─ 无需复杂切块
├─ 成本与 OpenAI small 相同
└─ 适合完整段落嵌入
```

---

## 六、模型评测方法

### 6.1 MTEB 排行榜

**MTEB**（Massive Text Embedding Benchmark）是最权威的 Embedding 模型评测基准。

**查看排行榜**：
https://huggingface.co/spaces/mteb/leaderboard

**主要指标**：
```
1. Classification（分类）：文本分类准确率
2. Clustering（聚类）：文本聚类质量
3. Pair Classification（对分类）：文本对关系判断
4. Reranking（重排序）：结果排序准确度
5. Retrieval（检索）：信息检索召回率
6. STS（语义相似度）：相似度计算精度
7. Summarization（摘要）：摘要质量
```

**RAG 最关注**：**Retrieval**（检索）指标

### 6.2 自定义评测

#### 评测流程

```python
# 1. 准备测试集
test_data = [
    {"query": "如何提高向量检索精度？", "expected_doc_id": 123},
    {"query": "RAG系统性能优化", "expected_doc_id": 456},
    # ... 更多测试用例
]

# 2. 对比多个模型
models = {
    "openai-small": OpenAIEmbedding("text-embedding-3-small"),
    "bge-large": SentenceTransformer("BAAI/bge-large-zh-v1.5"),
}

# 3. 计算召回率
for model_name, model in models.items():
    recall_at_1 = 0
    recall_at_5 = 0
    
    for test in test_data:
        query_vec = model.encode(test["query"])
        results = vector_db.search(query_vec, top_k=5)
        
        if results[0].id == test["expected_doc_id"]:
            recall_at_1 += 1
        
        if test["expected_doc_id"] in [r.id for r in results]:
            recall_at_5 += 1
    
    print(f"{model_name}:")
    print(f"  Recall@1: {recall_at_1 / len(test_data):.2%}")
    print(f"  Recall@5: {recall_at_5 / len(test_data):.2%}")
```

#### 评测指标

```
Recall@K（召回率@K）：
└─ 前K个结果中包含正确答案的比例

Precision@K（精确率@K）：
└─ 前K个结果中正确答案占的比例

MRR（Mean Reciprocal Rank）：
└─ 正确答案排名的倒数的平均值

NDCG（归一化折损累积增益）：
└─ 考虑排序的综合指标
```

---

## 七、实战建议

### 7.1 选型检查清单

```
☐ 确定主要语言（中文/英文/多语言）
☐ 评估文档数量（< 100万 / > 100万）
☐ 计算预算（月成本预期）
☐ 确认部署能力（API调用 / 自部署）
☐ 明确性能要求（极致精度 / 够用即可）
☐ 考虑文档长度（< 512 tokens / > 512 tokens）
☐ 评估项目周期（短期POC / 长期项目）
```

### 7.2 常见误区

❌ **误区1：一味追求 MTEB 高分**
```
错误想法：MTEB 分数越高越好
正确做法：在自己的数据上测试，MTEB 只是参考
```

❌ **误区2：只看价格不看效果**
```
错误想法：免费开源一定比付费好
正确做法：综合考虑部署成本、维护成本、人力成本
```

❌ **误区3：频繁更换模型**
```
错误想法：新模型出来就立刻切换
正确做法：模型切换成本极高（需重建所有索引），谨慎决策
```

❌ **误区4：忽略维度差异**
```
错误想法：维度越高越好
正确做法：维度高 = 内存占用高，需平衡性能和资源
```

### 7.3 迁移策略

如果需要从一个模型切换到另一个：

```python
# 1. 并行运行期（1-2周）
# 同时维护两个向量索引
old_index = load_index("faiss_openai_small.index")
new_index = load_index("faiss_bge_large.index")

# 2. 灰度切换期（1-2周）
# 按比例分流
if random.random() < 0.1:  # 10% 流量
    results = new_index.search(query)
else:
    results = old_index.search(query)

# 3. 全量切换
# 验证效果OK后，100% 切到新模型

# 4. 清理旧索引
# 删除旧模型和索引文件
```

### 7.4 微调建议

如果开源模型效果不理想，可以考虑微调：

```python
# 适合微调的模型：
├─ bge-large-zh-v1.5
├─ bge-m3
└─ gte-large-zh

# 微调数据准备：
train_data = [
    {"query": "如何提高检索精度？", "positive": "文档123", "negative": "文档456"},
    # ... 至少 1000-10000 个样本对
]

# 微调方法：
# 1. 使用 sentence-transformers 框架
# 2. 准备正负样本对
# 3. 在你的领域数据上微调
# 4. 评估效果提升

# 预期效果：
# 领域召回率提升 10-30%
```

详见：[Embedding 模型微调教程](./embedding_finetune.ipynb)

---

## 八、总结

### 快速推荐

| 你的情况 | 推荐模型 | 一句话理由 |
|---------|---------|-----------|
| 快速验证想法 | text-embedding-3-small | 性价比最高，快速上线 |
| 中文为主，长期项目 | bge-large-zh-v1.5 | 中文最好，自部署省钱 |
| 多语言，预算充足 | text-embedding-3-large | 综合效果最好 |
| 长文档（> 512 tokens） | JinaAI embeddings-v3 | 8K 上下文，无需切块 |
| 成本极度敏感 | bge-large-zh-v1.5 | 免费开源，自己部署 |

### 关键决策点

```
1. 语言：中文主导 → BGE，多语言 → OpenAI/JinaAI
2. 预算：充足 → API 调用，紧张 → 自部署
3. 周期：短期（< 6月）→ API，长期 → 自部署
4. 文档长度：长文档 → JinaAI v3，正常 → 其他
5. 精度要求：极致 → large 系列，一般 → small/中型
```

---

## 相关资源

### 文档
- [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard)
- [BGE 官方文档](https://github.com/FlagOpen/FlagEmbedding)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [JinaAI Docs](https://jina.ai/embeddings/)

### 实战代码
- [Embedding 模型对比实验](./embedding_compare.ipynb)
- [Embedding 模型微调教程](./embedding_finetune.ipynb)

---

💡 **最后建议**：不要纠结于"最好"的模型，而应该找到"最适合"你场景的模型。建议在自己的数据上做小规模测试（100-1000条），对比 2-3 个候选模型，选择召回率最高且成本可接受的方案！

**Happy Embedding! 🚀**

