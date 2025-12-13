# ko_tokenizer.py
from __future__ import annotations
from typing import List

from kiwipiepy import Kiwi

# 전역 1회 생성(속도)
_KIWI = Kiwi()

# 기본: 명사만 남김 (사과랑/사과의/사과같이 -> "사과"만 남게 됨)
_KEEP_TAGS = {"NNG", "NNP"}  # 일반명사, 고유명사

def tokenize_ko(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    tokens = _KIWI.tokenize(text, normalize_coda=True)

    out: List[str] = []
    for t in tokens:
        if t.tag in _KEEP_TAGS:
            out.append(t.form)

    return out
