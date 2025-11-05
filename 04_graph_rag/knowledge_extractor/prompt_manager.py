"""提示词管理器 - 支持业务逻辑的灵活配置"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptManager:
    """提示词管理器
    
    管理知识抽取相关的提示词配置，支持根据不同业务场景灵活调整。
    """

    def __init__(
        self,
        prompts: Optional[Dict[str, Any]] = None,
        prompt_file_path: Optional[str] = None,
    ):
        """初始化提示词管理器
        
        Args:
            prompts: 提示词字典，如果提供则直接使用
            prompt_file_path: 提示词文件路径，如果提供则从文件加载
        """
        if prompts:
            self.prompts = prompts
        elif prompt_file_path:
            self.prompts = self._load_prompts_from_file(prompt_file_path)
        else:
            # 默认从项目标准位置加载
            self.prompts = self._load_default_prompts()

        # 验证必需的提示词键
        self._validate_prompts()

    def _load_default_prompts(self) -> Dict[str, Any]:
        """加载默认提示词"""
        try:
            # 尝试从标准位置导入
            from app.prompts.knowledge_graph_prompt import PROMPTS
            return PROMPTS
        except ImportError:
            # 如果导入失败，尝试从当前目录加载
            try:
                import sys
                import os
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                prompt_file = os.path.join(current_dir, "knowledge_graph_prompt.py")
                if os.path.exists(prompt_file):
                    return self._load_prompts_from_file(prompt_file)
            except Exception as e:
                logger.warning(f"无法加载默认提示词: {str(e)}")
                return self._get_fallback_prompts()

    def _load_prompts_from_file(self, file_path: str) -> Dict[str, Any]:
        """从文件加载提示词
        
        Args:
            file_path: 提示词文件路径（.py文件）
            
        Returns:
            Dict: 提示词字典
        """
        try:
            # 动态导入Python文件
            import importlib.util
            spec = importlib.util.spec_from_file_location("prompts_module", file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载提示词文件: {file_path}")
            
            prompts_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(prompts_module)
            
            if hasattr(prompts_module, "PROMPTS"):
                return prompts_module.PROMPTS
            else:
                raise ValueError(f"提示词文件中未找到 PROMPTS 变量")
                
        except Exception as e:
            logger.error(f"加载提示词文件失败 {file_path}: {str(e)}")
            raise

    def _validate_prompts(self):
        """验证提示词是否包含必需的键"""
        required_keys = [
            "DEFAULT_LANGUAGE",
            "DEFAULT_TUPLE_DELIMITER",
            "DEFAULT_RECORD_DELIMITER",
            "DEFAULT_COMPLETION_DELIMITER",
            "DEFAULT_ENTITY_TYPES",
            "entity_extraction",
            "entity_extraction_examples",
        ]
        
        missing_keys = [key for key in required_keys if key not in self.prompts]
        if missing_keys:
            logger.warning(f"提示词缺少以下键: {missing_keys}")

    def _get_fallback_prompts(self) -> Dict[str, Any]:
        """获取备用提示词（最小化配置）"""
        return {
            "DEFAULT_LANGUAGE": "中文",
            "DEFAULT_TUPLE_DELIMITER": "<|>",
            "DEFAULT_RECORD_DELIMITER": "##",
            "DEFAULT_COMPLETION_DELIMITER": "<|COMPLETE|>",
            "DEFAULT_ENTITY_TYPES": ["实体"],
            "entity_extraction": "请从以下文本中提取实体和关系：\n{input_text}",
            "entity_extraction_examples": [],
        }

    def get_prompt(self, key: str, **kwargs) -> str:
        """获取格式化的提示词
        
        Args:
            key: 提示词键名
            **kwargs: 格式化参数
            
        Returns:
            str: 格式化后的提示词
        """
        if key not in self.prompts:
            raise KeyError(f"提示词键 '{key}' 不存在")
        
        prompt_template = self.prompts[key]
        
        if isinstance(prompt_template, str):
            # 格式化字符串模板
            return prompt_template.format(**kwargs)
        elif isinstance(prompt_template, list):
            # 如果是列表（如examples），返回格式化的列表
            return "\n".join([
                item.format(**kwargs) if isinstance(item, str) else str(item)
                for item in prompt_template
            ])
        else:
            return str(prompt_template)

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值
        """
        return self.prompts.get(key, default)

    def update_prompt(self, key: str, value: Any):
        """更新提示词
        
        Args:
            key: 提示词键名
            value: 新的提示词值
        """
        self.prompts[key] = value
        logger.info(f"已更新提示词: {key}")

    def update_config(self, key: str, value: Any):
        """更新配置
        
        Args:
            key: 配置键名
            value: 新的配置值
        """
        self.prompts[key] = value
        logger.info(f"已更新配置: {key}")

    @property
    def language(self) -> str:
        """获取语言配置"""
        return self.prompts.get("DEFAULT_LANGUAGE", "中文")

    @property
    def tuple_delimiter(self) -> str:
        """获取元组分隔符"""
        return self.prompts.get("DEFAULT_TUPLE_DELIMITER", "<|>")

    @property
    def record_delimiter(self) -> str:
        """获取记录分隔符"""
        return self.prompts.get("DEFAULT_RECORD_DELIMITER", "##")

    @property
    def completion_delimiter(self) -> str:
        """获取完成分隔符"""
        return self.prompts.get("DEFAULT_COMPLETION_DELIMITER", "<|COMPLETE|>")

    @property
    def entity_types(self) -> List[str]:
        """获取实体类型列表"""
        entity_types = self.prompts.get("DEFAULT_ENTITY_TYPES", [])
        if isinstance(entity_types, list):
            return entity_types
        elif isinstance(entity_types, str):
            return [t.strip() for t in entity_types.split(",")]
        else:
            return []

    def get_examples(self) -> List[str]:
        """获取示例列表"""
        examples = self.prompts.get("entity_extraction_examples", [])
        if isinstance(examples, list):
            return examples
        else:
            return []

