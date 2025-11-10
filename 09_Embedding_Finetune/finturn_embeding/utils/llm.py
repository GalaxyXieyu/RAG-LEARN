"""
统一的 LLM 调用与常用模板封装

提供：
- chat_complete: OpenAI 兼容接口的基础封装（重试/超时）
- single_choice: 在指定 labels 中做单选分类
- judge_factor_relevance: 因子相关性判定（positive/negative）
"""
from __future__ import annotations

from typing import Dict, List, Optional
import time
import requests


def chat_complete(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int = 30,
    retries: int = 2,
    temperature: float = 0.0,
    top_p: float = 0.1,
    max_tokens: int = 64,
) -> Optional[str]:
    """最小封装的 Chat Completions 请求，返回 content 文本或 None。"""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
        except Exception:
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
    return None


def single_choice(
    labels: List[str],
    context: Dict[str, str],
    base_url: str,
    api_key: str,
    model: str,
    timeout: int = 20,
    retries: int = 2,
) -> Optional[str]:
    """在 labels 中单选一个标签，返回命中或 None。"""
    sys_prompt = (
        "你是一个严谨的文本分类器。请只从以下标签中选择且仅选择一个："
        + ",".join(labels)
        + "。返回纯文本标签，不要解释。"
    )
    # 合并上下文
    ctx_str = "\n".join([f"{k}={v}" for k, v in context.items() if v])
    user_prompt = f"请判定以下片段属于哪一类（单选）：\n{ctx_str}"

    content = chat_complete(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        timeout=timeout,
        retries=retries,
        max_tokens=16,
    )
    if not content:
        return None
    content = content.replace("阶段", "").strip()
    for lab in labels:
        if lab in content or content == lab:
            return lab
    return None


def judge_factor_relevance(
    query: str,
    content: str,
    headers_norm: str = "",
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    timeout: int = 30,
    retries: int = 2,
) -> Optional[str]:
    """
    使用 LLM 判断 content 是否包含 query（因子）相关信息。
    返回值："positive" / "negative" / None
    """
    if not (base_url and api_key and model):
        return None
    prompt_sys = "你是一个专业的工程数据标注助手，擅长分析工程清单表格数据。"
    table_context = f"\n表格列名：{headers_norm}\n" if headers_norm else ""
    prompt_user = (
        f"查询因子：{query}\n"
        f"{table_context}"
        f"文本内容（可能是表格数据，注意列对齐可能有问题）：\n{content[:2500]}\n\n"
        f"请仔细判断上述内容是否包含关于「{query}」这个因子的具体数据。\n\n"
        f"只回答 positive 或 negative，不要有其他内容。"
    )

    result = chat_complete(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt_user}],
        timeout=timeout,
        retries=retries,
        max_tokens=10,
    )
    if not result:
        return None
    r = result.strip().lower()
    if "positive" in r:
        return "positive"
    if "negative" in r:
        return "negative"
    return None
