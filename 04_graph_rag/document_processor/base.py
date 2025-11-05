"""文档处理器基础接口"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DocumentProcessorInterface(ABC):
    """文档处理器抽象接口"""

    @abstractmethod
    def process(self, input_data: Any) -> List[Dict[str, Any]]:
        """处理文档
        
        Args:
            input_data: 输入数据（文件路径、文本等）
            
        Returns:
            List[Dict]: 处理后的chunk列表
        """
        pass

