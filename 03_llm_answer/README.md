# 第三步：LLM答案生成 🎯

## 模块概述

有了检索到的相关内容，如何让LLM生成高质量的答案？本模块介绍Prompt工程、答案优化等技巧，这是构建RAG Agent的基础。

## 学习目标

- ✅ 掌握基础RAG的Prompt模板设计
- ✅ 学会优化答案质量和格式
- ✅ 理解如何标注引用来源
- ✅ 避免LLM"幻觉"和编造

## 基础RAG流程

```python
# 标准RAG流程
query = "什么是RAG？"

# 1. 检索相关文档
retrieved_docs = retriever.get_relevant_documents(query)

# 2. 构建Prompt
context = "\n".join([doc.page_content for doc in retrieved_docs])
prompt = f"""
根据以下参考资料回答问题：

参考资料：
{context}

问题：{query}

请基于参考资料给出准确的回答。如果参考资料中没有相关信息，请说"抱歉，我无法回答这个问题"。
"""

# 3. LLM生成答案
answer = llm.invoke(prompt)
```

## Prompt工程技巧

### 1. 基础模板

```python
basic_template = """
你是一个专业的AI助手。请根据以下参考资料回答用户的问题。

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 必须基于参考资料回答
2. 如果资料中没有相关信息，明确告知用户
3. 回答要准确、简洁
"""
```

### 2. 带格式控制的模板

```python
structured_template = """
你是一个严谨的客服机器人。

【知识库内容】
{context}

【用户问题】
{question}

【回答规范】
1. 回答格式：
   - 首先给出核心答案（1-2句话）
   - 如有必要，补充详细解释
   - 最后提供相关建议（可选）

2. 语气要求：
   - 专业但友好
   - 不使用过于技术化的术语
   - 用"您"称呼用户

3. 准确性要求：
   - 只使用知识库中的信息
   - 不确定的内容要明确说明
   - 提供信息来源（如文档名称）

请严格按照以上规范回答。
"""
```

### 3. 带示例的Few-shot模板

```python
fewshot_template = """
你是一个产品推荐助手。请根据知识库推荐产品。

【推荐示例】
问题：我想买一个性价比高的手机
回答：根据您的需求，我为您推荐以下产品：
- **小米13**：价格2999元，搭载骁龙8 Gen2处理器，性价比出色
- **Redmi K60**：价格1999元，性能强劲，适合游戏用户
这两款产品在同价位中配置领先，用户评价优秀。

【知识库】
{context}

【用户问题】
{question}

请参考示例格式回答。
"""
```

## 答案质量优化

### 1. 控制答案长度

```python
# 方法1：在Prompt中明确要求
"请用50字以内简要回答"
"请详细说明，至少300字"

# 方法2：使用LLM参数
llm = ChatOpenAI(
    temperature=0,  # 降低随机性
    max_tokens=200  # 限制输出长度
)
```

### 2. 优化答案结构

```python
structured_prompt = """
请按照以下结构回答：

## 核心答案
[用1-2句话直接回答问题]

## 详细说明
[提供必要的背景和细节]

## 注意事项
[如有特殊情况需要说明]

## 参考来源
[标注信息来源]
"""
```

### 3. 多语言支持

```python
multilingual_prompt = """
请用{language}回答以下问题。

Context: {context}
Question: {question}

Requirements:
- Answer in {language}
- Keep professional tone
- Be concise and accurate
"""
```

## 引用来源标注

让用户知道答案的来源，增强可信度：

```python
citation_template = """
请根据以下文档回答问题，并在答案中标注来源。

文档列表：
{context_with_ids}

问题：{question}

回答格式：
[你的回答内容] [来源：文档1]

示例：
RAG是检索增强生成技术 [来源：文档2]，它可以提升LLM的准确性 [来源：文档1，文档3]
"""

# 构建带ID的上下文
def format_context_with_ids(docs):
    context = ""
    for i, doc in enumerate(docs, 1):
        context += f"\n[文档{i}] {doc.page_content}\n"
        context += f"来源：{doc.metadata.get('source', '未知')}\n"
    return context
```

## 避免幻觉和编造

### 策略1：明确约束

```python
no_hallucination_prompt = """
严格要求：
1. 只能使用下面【参考资料】中的信息回答
2. 如果参考资料中没有相关信息，必须回复："抱歉，提供的资料中没有相关信息"
3. 绝对不允许根据你的知识库回答
4. 不允许推测或猜测

【参考资料】
{context}

【问题】
{question}
"""
```

### 策略2：答案验证

```python
# 两阶段验证
# 阶段1：生成答案
answer = llm.invoke(answer_prompt)

# 阶段2：验证答案是否基于参考资料
verification_prompt = f"""
请判断以下答案是否完全基于参考资料。

参考资料：
{context}

生成的答案：
{answer}

判断标准：
- 答案中的每个事实是否都能在参考资料中找到
- 是否包含参考资料之外的信息

请回答：是/否，并说明理由。
"""

verification = llm.invoke(verification_prompt)
```

### 策略3：置信度评分

```python
confidence_prompt = """
根据参考资料回答问题，并给出置信度评分（0-100）。

参考资料：
{context}

问题：{question}

回答格式：
答案：[你的回答]
置信度：[0-100的分数]
理由：[为什么给这个置信度分数]

评分标准：
- 90-100：参考资料中有明确的完整答案
- 70-89：参考资料中有相关信息，但需要推理
- 50-69：参考资料中只有部分相关信息
- 0-49：参考资料中基本没有相关信息
"""
```

## 不同场景的Prompt示例

### 场景1：客服问答

```python
customer_service_prompt = """
你是{company}的智能客服。

知识库：{context}
客户问题：{question}

回答要求：
1. 称呼客户为"您"
2. 语气亲切、专业
3. 如果知识库中没有信息，引导客户联系人工客服
4. 提供清晰的操作步骤（如适用）

回答格式：
您好！[回答内容]

如有其他问题，欢迎随时咨询～
"""
```

### 场景2：技术文档查询

```python
tech_doc_prompt = """
你是一个技术文档助手。

相关文档：
{context}

技术问题：{question}

回答要求：
1. 使用技术专业术语
2. 提供代码示例（如适用）
3. 标注API版本或文档版本
4. 如有多种解决方案，列出并比较

请给出准确的技术回答。
"""
```

### 场景3：教育辅导

```python
education_prompt = """
你是一个耐心的教师助手。

教材内容：
{context}

学生问题：{question}

教学要求：
1. 用简单易懂的语言解释
2. 提供具体例子帮助理解
3. 鼓励学生思考
4. 如果是超纲内容，建议学生稍后学习

请用教学的方式回答。
"""
```

## 实践练习

### 📓 Notebook：llm_answer.ipynb

本模块包含：

1. **基础RAG实现**
   - 完整的检索+生成流程
   - 不同Prompt模板对比

2. **答案优化实验**
   - 格式控制
   - 风格调整
   - 长度控制

3. **防幻觉策略**
   - 约束性Prompt
   - 答案验证
   - 置信度评分

4. **实际案例**
   - 客服场景
   - 技术文档
   - 教育辅导

## 调试技巧

### 1. Prompt调试

```python
# 打印实际发送给LLM的完整Prompt
print("=" * 50)
print("实际Prompt:")
print(prompt)
print("=" * 50)

# 对比不同Prompt的效果
prompts = [prompt_v1, prompt_v2, prompt_v3]
for i, p in enumerate(prompts, 1):
    answer = llm.invoke(p)
    print(f"版本{i}回答：{answer}")
```

### 2. 温度参数调优

```python
# temperature影响答案的创造性
temperatures = [0, 0.3, 0.7, 1.0]
for temp in temperatures:
    llm = ChatOpenAI(temperature=temp)
    answer = llm.invoke(prompt)
    print(f"Temperature={temp}: {answer}")

# 建议值：
# - temperature=0: 客服、技术文档（需要准确性）
# - temperature=0.3-0.7: 内容创作、头脑风暴
```

## 常见问题

### Q1: LLM总是输出很长的答案怎么办？
**A**: 
1. 在Prompt中明确字数限制
2. 使用max_tokens参数
3. 加入示例展示期望的长度

### Q2: 如何让回答更口语化/更正式？
**A**: 
在Prompt中明确风格要求，并提供示例：
```python
"请用轻松口语化的方式回答，像朋友聊天一样"
"请用正式的书面语回答，适合商务场合"
```

### Q3: LLM不遵循Prompt指令怎么办？
**A**: 
1. 把重要指令放在开头和结尾
2. 使用"必须"、"严格"等强调词
3. 提供正反例
4. 考虑换更强的模型（如GPT-4）

## 下一步

掌握基础答案生成后，学习更高级的RAG系统：

➡️ [第四步：RAG智能体](../04_rag_agent/README.md)

---

💡 **提示**：Prompt工程是一门艺术，需要不断迭代优化。建议保存好用的Prompt模板，建立自己的Prompt库！

