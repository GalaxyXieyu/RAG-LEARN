"""
文件操作辅助函数
"""
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.minio_service import MinIOService
from finturn_embeding.config import REPORTS_DIR


def download_file_from_minio(
    bucket_name: str,
    object_key: str,
    target_path: Path
) -> bool:
    """
    从 MinIO 下载文件到本地
    
    Args:
        bucket_name: bucket 名称
        object_key: 对象键
        target_path: 目标文件路径
        
    Returns:
        是否成功
    """
    try:
        minio_service = MinIOService()
        result = minio_service.download_file(
            bucket_name=bucket_name,
            object_name=object_key,
            file_path=str(target_path)
        )
        return result.get("success", False)
    except Exception as e:
        print(f"从 MinIO 下载文件失败: {e}")
        return False


def copy_file_from_local(
    source_path: Path,
    target_path: Path
) -> bool:
    """
    从本地路径复制文件
    
    Args:
        source_path: 源文件路径
        target_path: 目标文件路径
        
    Returns:
        是否成功
    """
    try:
        if not source_path.exists():
            return False
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        return True
    except Exception as e:
        print(f"复制文件失败: {e}")
        return False


def save_manifest(manifest_data: Dict[str, Any], manifest_path: Path):
    """
    保存 manifest 文件
    
    Args:
        manifest_data: manifest 数据
        manifest_path: manifest 文件路径
    """
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存 manifest 失败: {e}")


def load_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    """
    加载 manifest 文件
    
    Args:
        manifest_path: manifest 文件路径
        
    Returns:
        manifest 数据，如果文件不存在返回 None
    """
    try:
        if not manifest_path.exists():
            return None
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载 manifest 失败: {e}")
        return None

