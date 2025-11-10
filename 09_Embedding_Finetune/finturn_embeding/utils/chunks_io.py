"""
chunks JSON 读写与 content 反查工具

规范化 chunk_id: {document_id}-{page_idx}-{chunk_index}
提供：
- parse_chunk_id
- load_chunks_json(document_id, chunks_dir)
- lookup_content(chunk_id, document_id, chunks_dir)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from functools import lru_cache


def parse_chunk_id(chunk_id: str) -> Optional[Tuple[str, int, int]]:
    try:
        parts = (chunk_id or "").split("-")
        if len(parts) < 3:
            return None
        document_id = parts[0]
        page_idx = int(parts[1])
        chunk_index = int(parts[2])
        return document_id, page_idx, chunk_index
    except Exception:
        return None


@lru_cache(maxsize=256)
def _load_json_cached(json_path_str: str) -> Optional[Dict]:
    json_path = Path(json_path_str)
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_chunks_json(document_id: str, chunks_dir: Path) -> Optional[Dict]:
    json_file = Path(chunks_dir) / f"{document_id}.json"
    return _load_json_cached(str(json_file))


def lookup_content(chunk_id: str, document_id: str, chunks_dir: Path) -> Optional[str]:
    data = load_chunks_json(document_id, chunks_dir)
    if not data:
        return None
    chunks = data.get("chunks", [])
    parsed = parse_chunk_id(chunk_id)
    if not parsed:
        return None
    _doc, page_idx, chunk_index = parsed
    for c in chunks:
        try:
            if int(c.get("page_idx", -1)) == page_idx and int(c.get("chunk_index", -1)) == chunk_index:
                return c.get("content") or ""
        except Exception:
            continue
    return None
