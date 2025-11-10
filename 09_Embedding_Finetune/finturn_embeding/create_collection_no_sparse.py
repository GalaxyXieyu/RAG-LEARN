"""
创建 Milvus 集合（无稀疏向量）- Qwen3-Embedding-0.6B (1024维)
"""
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 连接 Milvus
connections.connect(host="127.0.0.1", port="19530")

collection_name = "projects_documents_chunks_v2"

# Qwen3-Embedding-0.6B 的向量维度
VECTOR_DIM = 1024

# 删除旧集合（如果存在）
if utility.has_collection(collection_name):
    print(f"删除旧集合: {collection_name}")
    utility.drop_collection(collection_name)

# 定义字段（无 sparse_vector）
fields = [
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="project_code", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="page_idx", dtype=DataType.INT64),
    FieldSchema(name="block_type", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="stage", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="table_group_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="headers_norm", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
]

# 创建 schema
schema = CollectionSchema(
    fields=fields,
    description="文档块集合（纯稠密向量，Qwen3-Embedding-0.6B 1024维）",
    enable_dynamic_field=True
)

# 创建集合
collection = Collection(name=collection_name, schema=schema)

# 创建索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",  # 内积
    "params": {"nlist": 1024}
}
collection.create_index(field_name="dense_vector", index_params=index_params)

print(f"✅ 集合创建成功: {collection_name}")
print(f"   字段数: {len(fields)} (无 sparse_vector)")
print(f"   dense_vector 维度: {VECTOR_DIM} (Qwen3-Embedding-0.6B)")
print(f"   索引: IVF_FLAT (IP)")

collection.load()
print(f"✅ 集合已加载，可以开始入库")
