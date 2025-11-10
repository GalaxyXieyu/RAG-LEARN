"""
通用本地嵌入模型封装

提供统一的 Qwen3/Sentence-Transformers 加载与编码接口，
避免在各脚本内重复实现设备选择、环境变量与批量编码逻辑。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import os


class Embedder:
    """轻量封装 SentenceTransformer 的编码接口。"""

    def __init__(self, st_model):
        self.model = st_model

    def encode(self, texts: List[str], batch_size: int = 16, max_length: int = 512) -> List[List[float]]:
        """批量编码文本，返回归一化后的向量列表（list[list[float]]）。"""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # 转为 Python list 以便 JSON/序列化
        return vecs.tolist()


def get_embedder(model_dir: Path, cuda_device: Optional[int] = None) -> Embedder:
    """
    加载本地 Sentence-Transformers 格式的 embedding 模型并返回 Embedder。

    Args:
        model_dir: 模型目录
        cuda_device: 可选的 GPU 设备编号，例如 0/1/2

    Returns:
        Embedder 实例
    """
    # 降低无关依赖的干扰
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TRANSFORMERS_IMAGE_TRANSFORMS", "0")

    from sentence_transformers import SentenceTransformer
    import torch

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    model = SentenceTransformer(str(model_dir), trust_remote_code=True)

    # 设备选择
    if torch.cuda.is_available():
        device = f"cuda:{cuda_device}" if cuda_device is not None else "cuda"
    else:
        device = "cpu"
    model.to(device)

    return Embedder(model)
