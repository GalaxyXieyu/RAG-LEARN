# 知识抽取模块 - 提示词配置指南

## 概述

知识抽取模块支持通过 `PromptManager` 灵活配置业务提示词，可以根据不同的业务场景（如建筑项目管理、医疗、金融等）调整实体类型、关系类型和抽取规则。

## 核心组件

### PromptManager

提示词管理器，负责加载和管理提示词配置：

```python
from knowledge_extractor import PromptManager

# 方式1: 使用默认提示词（从app.prompts.knowledge_graph_prompt导入）
prompt_manager = PromptManager()

# 方式2: 从文件加载自定义提示词
prompt_manager = PromptManager(
    prompt_file_path="/path/to/custom_prompts.py"
)

# 方式3: 直接传入提示词字典
custom_prompts = {
    "DEFAULT_LANGUAGE": "中文",
    "DEFAULT_ENTITY_TYPES": ["人员", "事件", "地点"],
    "entity_extraction": "自定义的实体抽取提示词...",
    # ...
}
prompt_manager = PromptManager(prompts=custom_prompts)
```

### KnowledgeExtractor

知识抽取器，支持传入 `PromptManager` 或提示词文件路径：

```python
from knowledge_extractor import KnowledgeExtractor

# 方式1: 使用默认提示词
extractor = KnowledgeExtractor()

# 方式2: 指定提示词文件路径
extractor = KnowledgeExtractor(
    prompt_file_path="/path/to/custom_prompts.py"
)

# 方式3: 传入PromptManager实例
from knowledge_extractor import PromptManager
prompt_manager = PromptManager(prompt_file_path="custom_prompts.py")
extractor = KnowledgeExtractor(prompt_manager=prompt_manager)
```

## 提示词文件格式

提示词文件是一个Python文件，包含 `PROMPTS` 字典：

```python
# custom_prompts.py
PROMPTS = {}

# 基础配置
PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# 实体类型（根据业务需求定义）
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "项目基本信息",
    "空间位置",
    "时间安排",
    "资源设施",
    "规范要求",
    "风险因素",
    "缓解措施",
    "关联项目"
]

# 实体抽取提示词模板
PROMPTS["entity_extraction"] = """---目标---
你是一名专业的...专家。你的目标是...
...
{language}
...
实体类型: {entity_types}
文本:
{input_text}
"""

# 示例列表
PROMPTS["entity_extraction_examples"] = [
    """示例 1:
...
""",
    # 更多示例...
]
```

## 在KGService中使用

```python
from graph_rag import KGService, GraphRAGConfig

# 方式1: 通过配置指定提示词文件
config = GraphRAGConfig(
    prompt_file_path="/path/to/custom_prompts.py"
)
kg_service = KGService(config=config)

# 方式2: 直接传入KnowledgeExtractor
from knowledge_extractor import KnowledgeExtractor, PromptManager

prompt_manager = PromptManager(
    prompt_file_path="/path/to/custom_prompts.py"
)
extractor = KnowledgeExtractor(prompt_manager=prompt_manager)
kg_service = KGService(knowledge_extractor=extractor)
```

## 动态调整提示词

可以在运行时动态调整提示词：

```python
from knowledge_extractor import PromptManager

prompt_manager = PromptManager()

# 更新实体类型
prompt_manager.update_config(
    "DEFAULT_ENTITY_TYPES",
    ["新类型1", "新类型2", "新类型3"]
)

# 更新提示词模板
prompt_manager.update_prompt(
    "entity_extraction",
    "新的提示词模板..."
)
```

## 业务场景示例

### 建筑项目管理（默认）

使用 `/data/xieyu/Teaching/RAG/04_graph_rag/knowledge_graph_prompt.py`：

- 实体类型：项目基本信息、空间位置、时间安排、资源设施、规范要求、风险因素、缓解措施、关联项目
- 关系类型：空间关系、时间关系、合规关系、资源关系
- 时间格式：标准化格式（YYYY-MM-DD等）

### 医疗场景

可以创建 `medical_prompts.py`：

```python
PROMPTS = {}
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "疾病", "症状", "药物", "检查", "治疗方案", "医生", "患者"
]
PROMPTS["entity_extraction"] = """作为医疗信息抽取专家...
"""
```

### 金融场景

可以创建 `finance_prompts.py`：

```python
PROMPTS = {}
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "公司", "产品", "交易", "风险", "监管要求", "客户"
]
PROMPTS["entity_extraction"] = """作为金融信息抽取专家...
"""
```

## 提示词模板变量

提示词模板支持以下变量：

- `{language}`: 输出语言
- `{entity_types}`: 实体类型列表（逗号分隔）
- `{examples}`: 示例文本
- `{input_text}`: 输入文本
- `{tuple_delimiter}`: 元组分隔符
- `{record_delimiter}`: 记录分隔符
- `{completion_delimiter}`: 完成分隔符

## 最佳实践

1. **业务特定提示词**: 为每个业务场景创建独立的提示词文件
2. **版本管理**: 将提示词文件纳入版本控制
3. **测试验证**: 调整提示词后，使用测试数据验证效果
4. **文档化**: 记录每个提示词的业务含义和使用场景
5. **模块化**: 不同业务场景的提示词可以放在不同目录

