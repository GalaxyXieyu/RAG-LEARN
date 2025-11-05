# 多模态嵌入与检索 🎨

> 突破文本局限，让AI理解图像、音频、视频

## 📖 章节概述

本章节探索**多模态嵌入（Multimodal Embedding）**技术，学习如何让AI同时理解和处理文本、图像、音频、视频等多种模态的数据。这是RAG技术的重要扩展方向，能够实现更丰富的应用场景。

## 🎯 学习目标

完成本章节学习后，你将能够：

- ✅ 理解多模态嵌入的核心原理
- ✅ 掌握CLIP、ImageBind等主流模型
- ✅ 实现以图搜图、以文搜图功能
- ✅ 构建多模态RAG系统
- ✅ 处理视频和音频检索场景

---

## 一、什么是多模态嵌入？

### 1.1 核心概念

**单模态 vs 多模态**：

```
传统嵌入（单模态）：
文本 → Text Encoder → 向量 [0.2, 0.5, ...]
图像 → Image Encoder → 向量 [0.8, 0.1, ...]
问题：两个向量在不同空间，无法直接比较

多模态嵌入：
文本 → Multimodal Encoder → 统一向量空间
图像 → Multimodal Encoder → 统一向量空间
优势：文本和图像向量可以直接比较相似度！
```

### 1.2 应用场景

```
1. 以图搜图 (Image-to-Image Search)
   上传一张产品图 → 找到相似产品

2. 以文搜图 (Text-to-Image Search)
   输入"红色的跑车" → 检索出所有红色跑车图片

3. 以图搜文 (Image-to-Text Search)
   上传场景图片 → 找到相关的文章和描述

4. 视频检索 (Video Search)
   输入"进球瞬间" → 检索出足球比赛中的进球片段

5. 音频检索 (Audio Search)
   哼一段旋律 → 找到完整歌曲

6. 多模态问答
   "这张图片里的人在做什么？" → 理解图片并回答

7. 跨模态生成
   文本 → 生成图像（DALL-E）
   图像 → 生成描述
```

---

## 二、主流多模态模型

### 2.1 CLIP（Contrastive Language-Image Pre-training）

**开发者**：OpenAI  
**特点**：文本-图像联合嵌入的开创性工作

**核心原理**：
```python
# CLIP的对比学习机制
文本："一只猫坐在沙发上"    → Text Encoder  → 向量A
图像：[猫坐在沙发上的照片]  → Image Encoder → 向量B

训练目标：
- 匹配的文本-图像对 → 向量距离近
- 不匹配的文本-图像对 → 向量距离远
```

**优势**：
- ✅ 零样本图像分类（不需要训练即可分类新类别）
- ✅ 文本-图像跨模态检索
- ✅ 开源，易于使用

**应用示例**：
```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 文本和图像编码
text_inputs = processor(text=["一只猫", "一只狗"], return_tensors="pt", padding=True)
image = Image.open("cat.jpg")
image_inputs = processor(images=image, return_tensors="pt")

# 获取嵌入
with torch.no_grad():
    text_embeddings = model.get_text_features(**text_inputs)
    image_embeddings = model.get_image_features(**image_inputs)

# 计算相似度
similarity = torch.cosine_similarity(text_embeddings, image_embeddings)
print(f"文本'一只猫'与图像的相似度: {similarity[0].item():.4f}")
```

### 2.2 ImageBind（One Embedding Space To Bind Them All）

**开发者**：Meta AI  
**特点**：支持6种模态的统一嵌入空间

**支持的模态**：
1. 文本（Text）
2. 图像（Image）
3. 音频（Audio）
4. 深度图（Depth）
5. 热成像（Thermal）
6. IMU数据（运动传感器）

**核心优势**：
```
任意模态 ↔ 任意模态的检索
例如：
- 声音 → 图像（听到海浪声 → 找到海滩图片）
- 文本 → 音频（"雷声" → 找到打雷的音频）
- 图像 → 深度图
```

### 2.3 其他重要模型

| 模型 | 开发者 | 模态支持 | 特点 |
|------|-------|----------|------|
| **ALIGN** | Google | 文本+图像 | 更大规模训练（18亿对） |
| **Florence** | Microsoft | 文本+图像 | 统一的视觉基础模型 |
| **Jina CLIP** | Jina AI | 文本+图像 | 支持多语言，中文友好 |
| **Chinese CLIP** | OFA-Sys | 文本+图像 | 专为中文优化 |
| **BEiT-3** | Microsoft | 文本+图像 | Vision-Language预训练 |

---

## 三、多模态RAG架构

### 3.1 传统RAG vs 多模态RAG

**传统文本RAG**：
```
用户提问(文本) → 文本检索 → 文本片段 → LLM → 文本答案
```

**多模态RAG**：
```
用户提问(文本/图像/语音) 
    ↓
多模态检索(文本+图像+视频+音频)
    ↓
多模态内容(文本描述 + 相关图片 + 视频片段)
    ↓
多模态LLM (GPT-4V, LLaVA, Qwen-VL)
    ↓
富文本答案(文字 + 图片 + 链接)
```

### 3.2 系统架构示例

```python
# 多模态RAG系统伪代码

class MultimodalRAG:
    def __init__(self):
        self.text_embedder = OpenAIEmbedding()
        self.image_embedder = CLIPModel()
        self.video_embedder = VideoEmbedder()
        self.vector_db = MilvusClient()
        self.multimodal_llm = GPT4V()
    
    def index_document(self, doc):
        """索引多模态文档"""
        # 1. 提取文本内容
        text_chunks = self.chunk_text(doc.text)
        text_vectors = self.text_embedder.encode(text_chunks)
        
        # 2. 提取图像
        images = self.extract_images(doc)
        image_vectors = self.image_embedder.encode(images)
        
        # 3. 提取视频关键帧
        if doc.has_video():
            frames = self.extract_key_frames(doc.video)
            frame_vectors = self.video_embedder.encode(frames)
        
        # 4. 存入向量库（不同Collection或用标签区分）
        self.vector_db.insert("text_collection", text_vectors, metadata=...)
        self.vector_db.insert("image_collection", image_vectors, metadata=...)
    
    def search(self, query, query_type="text"):
        """多模态检索"""
        if query_type == "text":
            # 文本查询 → 检索文本 + 图像
            text_results = self.search_text(query)
            image_results = self.search_images_by_text(query)
        
        elif query_type == "image":
            # 图像查询 → 检索相似图像 + 相关文本
            image_results = self.search_images_by_image(query)
            text_results = self.search_text_by_image(query)
        
        return self.merge_results(text_results, image_results)
    
    def generate_answer(self, query, retrieved_contents):
        """多模态答案生成"""
        # 构建多模态prompt
        prompt = {
            "text": query,
            "images": [content.image for content in retrieved_contents if content.has_image()],
            "context": [content.text for content in retrieved_contents]
        }
        
        # 调用多模态LLM
        answer = self.multimodal_llm.generate(prompt)
        return answer
```

---

## 四、实战应用场景

### 4.1 电商产品搜索

**需求**：用户上传一张衣服照片，找到相似的在售商品

**实现方案**：
```python
# 1. 商品库索引
products = load_products()  # 包含图片和描述
for product in products:
    # 图像嵌入
    image_embedding = clip_model.encode_image(product.image)
    # 文本嵌入
    text_embedding = clip_model.encode_text(product.description)
    # 存储
    vector_db.insert({
        "product_id": product.id,
        "image_vector": image_embedding,
        "text_vector": text_embedding,
        "price": product.price,
        "category": product.category
    })

# 2. 用户搜索
user_image = upload_image()
query_vector = clip_model.encode_image(user_image)

# 3. 检索相似商品
results = vector_db.search(
    collection="products",
    query_vector=query_vector,
    top_k=10,
    filter="price < 500 AND category == 'clothing'"
)

# 4. 展示结果
for result in results:
    print(f"商品: {result.name}, 相似度: {result.score:.2%}")
    display(result.image)
```

### 4.2 智能相册管理

**需求**：通过文本描述找到相册中的照片

**示例查询**：
- "我和小明在海边的合影"
- "2023年春节的照片"
- "有小狗的照片"
- "夕阳的风景照"

**实现要点**：
```python
# 1. 照片预处理
for photo in album:
    # 提取图像特征
    image_vector = clip_model.encode_image(photo)
    
    # 提取元数据
    metadata = {
        "date": photo.exif_data.date,
        "location": photo.exif_data.gps,
        "people": detect_faces(photo),  # 人脸识别
        "objects": detect_objects(photo),  # 物体检测
    }
    
    # 存储
    save_to_vector_db(image_vector, metadata)

# 2. 文本搜索
query = "我和小明在海边的合影"
query_vector = clip_model.encode_text(query)
results = vector_db.search(query_vector, filter="people CONTAINS '小明'")
```

### 4.3 视频内容检索

**需求**：在长视频中找到特定场景

**实现方案**：
```python
# 1. 视频预处理
video = load_video("lecture.mp4")
frames = extract_frames(video, fps=1)  # 每秒提取1帧

for i, frame in enumerate(frames):
    # 帧嵌入
    frame_vector = clip_model.encode_image(frame)
    
    # 如果有字幕，也编码
    if has_subtitle(video, timestamp=i):
        subtitle_text = get_subtitle(video, i)
        text_vector = clip_model.encode_text(subtitle_text)
    
    # 存储（带时间戳）
    vector_db.insert({
        "video_id": video.id,
        "timestamp": i,
        "frame_vector": frame_vector,
        "subtitle_vector": text_vector
    })

# 2. 场景搜索
query = "讲到向量数据库的部分"
query_vector = clip_model.encode_text(query)
results = vector_db.search(query_vector, filter="video_id == 'lecture'")

# 3. 定位时间点
for result in results:
    print(f"找到相关场景: {result.timestamp}秒")
    video.seek(result.timestamp)
```

### 4.4 医学影像检索

**需求**：根据病症描述找到相似的医学影像

**应用价值**：
- 辅助诊断（找到相似病例）
- 医学教学（根据描述找示例）
- 病例研究

**注意事项**：
- 需要专门的医学多模态模型（如BiomedCLIP）
- 数据隐私和合规要求
- 需要医生审核

---

## 五、技术实现要点

### 5.1 向量数据库Schema设计

**多模态Collection设计**：

**方案1：统一Collection + 模态标签**
```python
schema = {
    "id": "varchar",
    "content_type": "varchar",  # "text", "image", "video", "audio"
    "vector": "float_vector(512)",  # 统一维度
    "text_content": "varchar",
    "image_url": "varchar",
    "metadata": "json"
}
```

**方案2：分离Collection**
```python
# Text Collection
text_schema = {
    "id": "varchar",
    "vector": "float_vector(1024)",
    "text": "varchar",
    "source_doc_id": "varchar"
}

# Image Collection
image_schema = {
    "id": "varchar",
    "vector": "float_vector(512)",
    "image_url": "varchar",
    "caption": "varchar",
    "source_doc_id": "varchar"
}
```

**推荐**：方案2（分离），理由：
- 不同模态可能用不同的Embedding模型（维度不同）
- 查询模式不同（纯文本 vs 纯图像 vs 混合）
- 索引类型优化不同

### 5.2 跨模态检索策略

**场景1：文本查询 → 多模态结果**
```python
def text_to_multimodal_search(query_text):
    # 文本嵌入
    text_vector = text_embedder.encode(query_text)
    
    # 并行检索
    text_results = text_collection.search(text_vector, top_k=10)
    
    # CLIP嵌入（跨模态）
    clip_vector = clip_model.encode_text(query_text)
    image_results = image_collection.search(clip_vector, top_k=5)
    
    # 合并结果
    return merge_results(text_results, image_results)
```

**场景2：图像查询 → 多模态结果**
```python
def image_to_multimodal_search(query_image):
    # 图像嵌入
    clip_vector = clip_model.encode_image(query_image)
    
    # 检索相似图像
    image_results = image_collection.search(clip_vector, top_k=10)
    
    # 检索相关文本（使用图像向量）
    text_results = text_collection.search(clip_vector, top_k=5)
    
    return merge_results(image_results, text_results)
```

### 5.3 性能优化

**1. 图像预处理缓存**
```python
# 避免重复编码
@lru_cache(maxsize=1000)
def get_image_embedding(image_path):
    image = load_image(image_path)
    return clip_model.encode(image)
```

**2. 批量编码**
```python
# 批量处理提升效率
images = load_images_batch(image_paths)
embeddings = clip_model.encode(images, batch_size=32)  # GPU加速
```

**3. 降维与量化**
```python
# 原始向量：512维 float32 → 2KB
# PQ压缩后：64字节 → 节省97%空间
from faiss import IndexPQ
index = IndexPQ(512, 64, 8)  # 压缩到64字节
```

---

## 六、实践教程（待开发）

### 📓 Notebook 1: CLIP图文检索入门
**文件**：`clip_basics.ipynb`（待开发）

**内容**：
1. CLIP模型加载与使用
2. 以文搜图实现
3. 以图搜文实现
4. 零样本图像分类
5. 向量数据库集成

### 📓 Notebook 2: 电商产品搜索系统
**文件**：`ecommerce_image_search.ipynb`（待开发）

**内容**：
1. 商品图像数据预处理
2. 多模态索引构建
3. 以图搜商品功能
4. 多条件过滤（价格、类别等）
5. 结果排序优化

### 📓 Notebook 3: 智能相册管理
**文件**：`photo_album_search.ipynb`（待开发）

**内容**：
1. 照片批量编码
2. 人脸识别集成
3. 自然语言照片搜索
4. 时间/地点过滤
5. Web UI 实现

### 📓 Notebook 4: 视频内容检索
**文件**：`video_search.ipynb`（待开发）

**内容**：
1. 视频关键帧提取
2. 帧级别索引构建
3. 场景检索与定位
4. 字幕联合检索
5. 时间轴可视化

### 📓 Notebook 5: 多模态RAG系统
**文件**：`multimodal_rag_system.ipynb`（待开发）

**内容**：
1. 多模态文档解析
2. 统一向量库设计
3. 跨模态检索实现
4. GPT-4V集成
5. 完整问答流程

---

## 七、模型选择指南

### 7.1 开源 vs 闭源

| 维度 | 开源模型（CLIP等） | 闭源API（OpenAI等） |
|------|-------------------|---------------------|
| **成本** | 免费，需自部署 | 按调用付费 |
| **性能** | 中等，持续改进 | 最好 |
| **定制性** | 可微调 | 无法微调 |
| **部署** | 需GPU | 直接调用 |
| **数据隐私** | 完全可控 | 上传到云端 |

### 7.2 模型推荐

**快速原型（POC）**：
```
推荐：OpenAI CLIP API 或 Jina AI
理由：快速验证，无需部署
```

**中文为主**：
```
推荐：Chinese CLIP 或 Jina CLIP
理由：中文效果优化，支持中文文本
```

**生产环境**：
```
推荐：自部署 CLIP + 向量数据库
理由：成本可控，性能稳定
```

**视频/音频**：
```
推荐：ImageBind（Meta）
理由：统一多模态空间，支持音视频
```

---

## 八、常见问题

### Q1: CLIP模型的向量维度可以修改吗？

**A**: 
```
不建议直接修改。CLIP的输出维度是固定的（如512维）。
如果需要降维，可以：
1. 使用PCA降维
2. 训练一个降维网络
3. 使用PQ量化（Faiss）

但降维会损失一定精度，需要权衡。
```

### Q2: 如何处理不同模态的相似度计算？

**A**:
```python
# 方法1：归一化后计算（推荐）
text_vec_normalized = text_vec / np.linalg.norm(text_vec)
image_vec_normalized = image_vec / np.linalg.norm(image_vec)
similarity = np.dot(text_vec_normalized, image_vec_normalized)

# 方法2：使用cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([text_vec], [image_vec])[0][0]
```

### Q3: 多模态检索的延迟如何优化？

**A**:
```
1. 图像预编码缓存（离线处理）
2. 批量编码（GPU加速）
3. 向量量化压缩（PQ/SQ）
4. 分层检索（粗筛 + 精排）
5. 使用更快的向量库（Milvus/Qdrant）
```

### Q4: 如何评估多模态检索效果？

**A**:
```
指标：
1. Recall@K：前K个结果中包含正确答案的比例
2. MRR（Mean Reciprocal Rank）：正确答案排名的倒数平均值
3. mAP（mean Average Precision）：平均精度均值
4. 用户满意度：实际使用反馈

方法：
- 构建测试集（查询-正确结果对）
- A/B测试不同模型
- 人工评估前10个结果的相关性
```

---

## 九、学习路线

### 路线1：快速入门（1-2天）

```
Day 1:
├─ 上午：阅读本README，理解多模态概念
├─ 下午：运行CLIP基础示例
└─ 晚上：实现简单的以图搜图

Day 2:
├─ 上午：集成向量数据库
├─ 下午：构建小型图片搜索系统
└─ 晚上：测试和优化
```

### 路线2：深入实践（1周）

```
Week 1:
├─ Day 1-2：CLIP模型深入学习
├─ Day 3：电商产品搜索实战
├─ Day 4：智能相册实现
├─ Day 5：视频检索探索
├─ Day 6-7：多模态RAG系统开发
```

### 路线3：项目应用（持续）

```
1. 选择实际应用场景
2. 收集和准备数据
3. 选择合适的模型
4. 部署向量数据库
5. 开发检索API
6. 前端界面开发
7. 性能测试与优化
8. 上线与迭代
```

---

## 十、相关资源

### 官方文档
- [CLIP GitHub](https://github.com/openai/CLIP)
- [ImageBind GitHub](https://github.com/facebookresearch/ImageBind)
- [Chinese CLIP](https://github.com/OFA-Sys/Chinese-CLIP)
- [Jina CLIP](https://jina.ai/embeddings/)

### 论文
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665)

### 在线Demo
- [CLIP Playground](https://huggingface.co/spaces/openai/clip)
- [ImageBind Demo](https://imagebind.metademolab.com/)

### 相关课程
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Multi-Modal Machine Learning (CMU)](https://cmu-multicomp-lab.github.io/mmml-course/fall2022/)

---

## 下一步

完成多模态嵌入学习后，可以：

➡️ 返回 [项目主页](../README.md)  
➡️ 探索 [GraphRAG图检索](../04_graph_rag/README.md)  
➡️ 学习 [企业级向量数据库](../07_vector_database_enterprise/README.md)

---

## 贡献指南

本章节正在持续完善中，欢迎贡献：

**待开发内容**：
- [ ] CLIP基础教程（clip_basics.ipynb）
- [ ] 电商产品搜索（ecommerce_image_search.ipynb）
- [ ] 智能相册管理（photo_album_search.ipynb）
- [ ] 视频内容检索（video_search.ipynb）
- [ ] 多模态RAG系统（multimodal_rag_system.ipynb）
- [ ] ImageBind多模态应用
- [ ] 音频检索实战
- [ ] 多模态模型微调

**如何贡献**：
1. Fork 项目
2. 创建特性分支
3. 提交代码和文档
4. 发起 Pull Request

---

💡 **寄语**：多模态是AI的未来方向！文本、图像、音频、视频的融合将带来更丰富的应用场景。虽然技术还在快速发展，但现在就是最好的学习时机。

**Let's explore the multimodal world! 🚀🎨🎵🎬**

