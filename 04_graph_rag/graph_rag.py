"""GraphRAG主入口文件 - 模块化版本"""

# 导入新的模块化组件
try:
    from .core.kg_service import KGService
    from .core.config import GraphRAGConfig
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    # 添加当前目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from core.kg_service import KGService
    from core.config import GraphRAGConfig

# 导出主要类和配置，保持API兼容
__all__ = ["KGService", "GraphRAGConfig"]
