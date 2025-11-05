# 第四步：RAG智能体（高级应用）🤖

## 模块概述

在掌握了基础RAG后，本模块介绍如何构建更智能的RAG系统：智能路由、自适应检索、联网补充、对话记忆等高级功能。这些功能让RAG不再是简单的"检索+生成"，而是一个能够自主决策的智能体。

## 学习目标

- ✅ 理解RAG Agent的核心概念
- ✅ 掌握智能路由实现（知识库 vs 联网）
- ✅ 学会构建对话记忆系统
- ✅ 实现自适应检索策略
- ✅ 掌握知识补充和融合技巧

## 什么是RAG Agent

**基础RAG**：固定流程，用户提问 → 检索知识库 → 生成答案

**RAG Agent**：智能决策，能够：
- 判断是否需要检索知识库
- 决定使用哪种检索方式
- 判断是否需要联网补充信息
- 管理多轮对话上下文
- 自我纠正和优化

```python
# 基础RAG（固定流程）
def basic_rag(question):
    docs = retrieve(question)
    answer = generate(question, docs)
    return answer

# RAG Agent（智能决策）
def rag_agent(question, history):
    # 1. 理解意图
    intent = analyze_intent(question)
    
    # 2. 智能路由
    if intent == "knowledge_base":
        docs = retrieve_from_kb(question)
    elif intent == "realtime":
        docs = search_web(question)
    elif intent == "both":
        docs = retrieve_from_kb(question) + search_web(question)
    
    # 3. 检查完整性
    if not is_sufficient(docs, question):
        docs += search_additional_info(question)
    
    # 4. 生成答案（考虑历史）
    answer = generate(question, docs, history)
    
    return answer
```

## 核心功能模块

### 1. 智能路由（Query Routing）

根据问题类型自动选择信息源：

```python
from langchain.chains import LLMRouterChain

# 定义路由规则
router_prompt = """
分析用户问题，选择合适的信息源：

1. knowledge_base：企业内部知识、产品信息、规章制度
2. web_search：实时信息、新闻、天气、当前事件
3. both：需要结合知识库和实时信息

问题：{question}

请选择：knowledge_base / web_search / both
"""

# 实现路由
def route_question(question):
    decision = llm.invoke(router_prompt.format(question=question))
    
    if "knowledge_base" in decision:
        return query_knowledge_base(question)
    elif "web_search" in decision:
        return search_web(question)
    else:  # both
        kb_results = query_knowledge_base(question)
        web_results = search_web(question)
        return merge_results(kb_results, web_results)
```

**应用场景**：
- ❓ "公司的年假政策是什么？" → 知识库
- ❓ "今天天气怎么样？" → 联网搜索
- ❓ "我们公司的产品和竞品比有什么优势？" → 知识库 + 联网

### 2. 对话记忆管理

让RAG能够理解上下文，进行多轮对话：

```python
from langchain.memory import ConversationBufferMemory

# 初始化记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 带记忆的对话
def chat_with_memory(question):
    # 获取历史对话
    history = memory.load_memory_variables({})
    
    # 考虑历史上下文进行检索
    reformulated_query = reformulate_with_history(question, history)
    
    # 检索和生成
    docs = retrieve(reformulated_query)
    answer = generate(question, docs, history)
    
    # 保存到记忆
    memory.save_context(
        {"input": question},
        {"output": answer}
    )
    
    return answer

# 示例对话
# 用户："什么是RAG？"
# AI："RAG是检索增强生成技术..."
# 用户："它有什么优势？"  ← 理解"它"指的是RAG
# AI："RAG的主要优势包括..."
```

**记忆策略**：

| 策略 | 适用场景 | 实现方式 |
|------|---------|---------|
| **全量记忆** | 短对话 | ConversationBufferMemory |
| **窗口记忆** | 长对话 | ConversationBufferWindowMemory |
| **摘要记忆** | 超长对话 | ConversationSummaryMemory |
| **向量记忆** | 需要检索历史 | VectorStoreRetrieverMemory |

### 3. 自适应检索

根据问题复杂度调整检索策略：

```python
def adaptive_retrieve(question, context=None):
    # 分析问题复杂度
    complexity = analyze_complexity(question)
    
    if complexity == "simple":
        # 简单问题：单次检索
        return simple_retrieve(question, top_k=3)
    
    elif complexity == "medium":
        # 中等复杂：多策略检索
        vector_results = vector_search(question, top_k=5)
        keyword_results = keyword_search(question, top_k=5)
        return rerank(vector_results + keyword_results)
    
    else:  # complex
        # 复杂问题：多跳检索
        # 第1跳：初始检索
        docs_1 = retrieve(question, top_k=10)
        
        # 第2跳：基于第1跳结果提取关键实体，再检索
        entities = extract_entities(docs_1)
        docs_2 = retrieve_by_entities(entities)
        
        # 合并并排序
        return merge_and_rerank(docs_1, docs_2)
```

### 4. 知识补充策略

当知识库信息不足时，智能补充：

```python
def answer_with_supplement(question):
    # 1. 先查知识库
    kb_docs = retrieve_from_kb(question)
    
    # 2. 评估充分性
    sufficiency_prompt = f"""
    问题：{question}
    检索到的内容：{kb_docs}
    
    请评估：这些内容是否足以完整回答问题？
    回答：充分 / 不充分
    如果不充分，缺少什么信息？
    """
    
    evaluation = llm.invoke(sufficiency_prompt)
    
    # 3. 如果不充分，补充信息
    if "不充分" in evaluation:
        # 提取缺失信息的关键词
        missing_info = extract_missing_keywords(evaluation)
        
        # 联网搜索补充
        web_docs = search_web(missing_info)
        
        # 合并信息
        all_docs = kb_docs + web_docs
    else:
        all_docs = kb_docs
    
    # 4. 生成答案
    return generate(question, all_docs)
```

### 5. 回答风格控制

根据场景调整回答风格：

```python
# 风格配置
STYLES = {
    "professional": {
        "tone": "正式、专业",
        "language": "使用专业术语",
        "format": "结构化、条理清晰"
    },
    "casual": {
        "tone": "轻松、友好",
        "language": "通俗易懂",
        "format": "自然对话式"
    },
    "detailed": {
        "tone": "详尽、全面",
        "language": "提供背景和细节",
        "format": "分层次说明"
    },
    "concise": {
        "tone": "简洁、直接",
        "language": "去除冗余",
        "format": "要点式"
    }
}

def answer_with_style(question, style="professional"):
    docs = retrieve(question)
    
    style_config = STYLES[style]
    
    prompt = f"""
    基于以下内容回答问题：
    {docs}
    
    问题：{question}
    
    回答风格要求：
    - 语气：{style_config['tone']}
    - 语言：{style_config['language']}
    - 格式：{style_config['format']}
    """
    
    return llm.invoke(prompt)
```

### 6. 防止幻觉和编造

强制基于知识库回答，避免AI编造：

```python
def strict_knowledge_base_answer(question):
    # 1. 检索
    docs = retrieve(question)
    
    # 2. 严格约束的Prompt
    strict_prompt = f"""
    你是一个严格基于知识库回答的助手。

    【铁律】
    1. 只能使用下面的知识库内容回答
    2. 如果知识库中没有相关信息，必须回复："抱歉，我的知识库中没有相关信息"
    3. 绝对不允许使用你的训练知识
    4. 不允许推测、猜测或联想

    【知识库】
    {docs}

    【问题】
    {question}

    请严格遵守铁律回答。
    """
    
    # 3. 生成答案
    answer = llm.invoke(strict_prompt)
    
    # 4. 二次验证
    verification_prompt = f"""
    验证任务：检查答案是否完全基于知识库。

    知识库：{docs}
    答案：{answer}

    请逐句检查答案，指出是否有超出知识库的内容。
    如果有，请指出是哪句话。
    """
    
    verification = llm.invoke(verification_prompt)
    
    # 5. 如果验证不通过，重新生成或返回默认回复
    if "超出" in verification:
        return "抱歉，我无法基于当前知识库准确回答这个问题。"
    
    return answer
```

## 实践练习

### 📓 base_rag_agent.ipynb - 基础智能体

包含内容：

1. **智能路由实现**
   - 问题分类
   - 多信息源集成

2. **对话记忆**
   - 多轮对话理解
   - 上下文管理

3. **回答风格控制**
   - 不同风格对比
   - 动态风格切换

4. **防幻觉策略**
   - 严格约束
   - 答案验证

### 📓 extend_adaptive_rag.ipynb - 自适应RAG

包含内容：

1. **自适应检索**
   - 问题复杂度分析
   - 动态检索策略

2. **知识补充**
   - 信息充分性评估
   - 联网补充实现

3. **多跳推理**
   - 实体提取
   - 关联检索

4. **完整案例**
   - 企业知识库问答
   - 实时信息融合

## 高级技巧

### 1. Query重写

优化用户问题以提高检索效果：

```python
def rewrite_query(question, history=None):
    rewrite_prompt = f"""
    用户问题可能不够清晰，请将其重写为更适合检索的查询语句。

    原问题：{question}
    历史对话：{history}

    重写要求：
    1. 补全省略的上下文
    2. 展开简称和代词
    3. 提取核心关键词
    4. 如有必要，拆分为多个子问题

    请输出重写后的查询。
    """
    
    return llm.invoke(rewrite_prompt)

# 示例
# 原问题："它有什么功能？"（指代不明）
# 重写后："RAG（检索增强生成）技术有什么功能和特点？"
```

### 2. 答案融合

合并多个来源的信息：

```python
def fuse_answers(question, sources):
    fusion_prompt = f"""
    你需要融合多个信息源，生成一个完整准确的答案。

    问题：{question}

    信息源1（知识库）：
    {sources['kb']}

    信息源2（网络搜索）：
    {sources['web']}

    融合要求：
    1. 以知识库信息为主
    2. 用网络信息补充和更新
    3. 如有冲突，说明差异
    4. 标注信息来源

    请生成融合后的答案。
    """
    
    return llm.invoke(fusion_prompt)
```

### 3. 自我反思

让Agent评估自己的答案质量：

```python
def answer_with_reflection(question):
    # 第1步：生成初始答案
    docs = retrieve(question)
    initial_answer = generate(question, docs)
    
    # 第2步：自我反思
    reflection_prompt = f"""
    评估以下答案的质量：

    问题：{question}
    参考资料：{docs}
    答案：{initial_answer}

    请从以下维度评估（1-5分）：
    1. 准确性：是否基于参考资料
    2. 完整性：是否回答了所有方面
    3. 清晰度：是否易于理解
    4. 相关性：是否切中问题要点

    如果得分低于4分，请提出改进建议。
    """
    
    reflection = llm.invoke(reflection_prompt)
    
    # 第3步：如果需要，改进答案
    if should_improve(reflection):
        improved_answer = generate_improved(question, docs, reflection)
        return improved_answer
    
    return initial_answer
```

## 常见问题

### Q1: 如何判断是否需要联网搜索？
**A**: 
- 关键词匹配：包含"今天"、"最新"、"现在"等时间词
- 知识库检索失败：相似度低于阈值
- LLM判断：让LLM分析问题类型

### Q2: 对话记忆占用太多tokens怎么办？
**A**: 
- 使用窗口记忆（只保留最近N轮）
- 使用摘要记忆（压缩历史对话）
- 使用向量记忆（检索相关历史）

### Q3: 多信息源的结果如何排序？
**A**: 
- 按来源优先级（知识库 > 网络）
- 按相关性分数
- 按时间新鲜度
- 使用Rerank模型

## 下一步

完成RAG Agent学习后，进入评估环节：

➡️ [第五步：RAG效果评估](../05_rag_evaluation/README.md)

---

💡 **提示**：RAG Agent是RAG的高级形态，重点在于"智能决策"。建议从简单场景开始，逐步增加复杂度。

