from __future__ import annotations
from typing import List
import re

from kiwipiepy import Kiwi

_kiwi = Kiwi()


_re_space = re.compile(r"\s+")

# Kiwi 품사 예: NNG(일반명사), NNP(고유명사), VV(동사), VA(형용사) ...
_KEEP_TAGS_NOUNS = {"NNG", "NNP"}

def tokenize_ko_nouns(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    tokens = _kiwi.tokenize(text, normalize_coda=True)

    out: List[str] = []
    for t in tokens:
        if t.tag in _KEEP_TAGS_NOUNS:
            # t.form이 표면형(토큰 문자열)
            if len(t.form) >= 1:
                out.append(t.form)

    return out
