from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfConfig:
    ngram_min: int = 1
    ngram_max: int = 1
    min_df: int = 1
    max_df: float = 1.0
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = False


class TfidfTrainer:
    def __init__(self, config: TfidfConfig | None = None):
        self.config = config or TfidfConfig()
        # token_pattern 설정을 이렇게 해두면 한글도 잘 잡힘
        self.vectorizer = TfidfVectorizer(
            ngram_range=(self.config.ngram_min, self.config.ngram_max),
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            use_idf=self.config.use_idf,
            smooth_idf=self.config.smooth_idf,
            sublinear_tf=self.config.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b",  # Unicode 단어 토큰 (한글 포함)
        )

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)

    def fit_transform(self, texts: List[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def to_export_dict(self) -> Dict[str, Any]:
        """C++ 쪽에서 사용할 수 있는 형태로 vocab/idf/config를 dict로 반환."""
        vocab = self.vectorizer.vocabulary_  # token -> index
        idf = self.vectorizer.idf_.tolist()

        return {
            "type": "tfidf",
            "config": asdict(self.config),
            "vocab": vocab,
            "idf": idf,
        }

    def save_json(self, path: str):
        data = self.to_export_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
