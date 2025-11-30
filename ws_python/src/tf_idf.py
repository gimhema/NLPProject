from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json
import numpy as np
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

    def load_json(self, path: str):
        """
        저장된 TF-IDF JSON을 로드하여 vectorizer 상태를 복원한다.

        JSON 구조 (save_json에서 만든 것):
        {
        "type": "tfidf",
        "config": { "ngram_min": .., "ngram_max": .., ... },
        "vocab": { "토큰": index, ... },
        "idf": [ ... ]
        }
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # config는 data["config"]에 있음
        cfg = data.get("config", {})
        # 안전하게 기본값 사용
        ngram_min = cfg.get("ngram_min", self.config.ngram_min)
        ngram_max = cfg.get("ngram_max", self.config.ngram_max)
        min_df = cfg.get("min_df", self.config.min_df)
        max_df = cfg.get("max_df", self.config.max_df)
        use_idf = cfg.get("use_idf", self.config.use_idf)
        smooth_idf = cfg.get("smooth_idf", self.config.smooth_idf)
        sublinear_tf = cfg.get("sublinear_tf", self.config.sublinear_tf)

        # vocab/idf 로드
        vocab = data.get("vocab", {})
        idf = data.get("idf", [])

        # 내부 상태 저장
        self.config.ngram_min = ngram_min
        self.config.ngram_max = ngram_max
        self.vocab = vocab
        self.idf = idf

        # vectorizer를 동일한 설정으로 재생성
        self.vectorizer = TfidfVectorizer(
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_df=max_df,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            token_pattern=r"(?u)\b\w+\b",
        )

        # vocabulary_ 와 idf_ 를 직접 주입 (transform 시 사용됨)
        # sklearn은 idf_를 numpy array로 기대하므로 변환
        try:
            # vocabulary: token -> index (정수)
            self.vectorizer.vocabulary_ = {k: int(v) for k, v in vocab.items()}
        except Exception:
            # 안전하게 그대로 할당
            self.vectorizer.vocabulary_ = vocab

        if idf:
            # idf는 numpy array로 주입
            self.vectorizer.idf_ = np.asarray(idf, dtype=float)
        else:
            # idf 정보가 없으면 기본 동작으로 두기 (transform만 쓰려면 불가)
            self.vectorizer.idf_ = np.array([])

        # 역방향 맵(필요시)
        self.inv_vocab = {int(idx): term for term, idx in self.vectorizer.vocabulary_.items()}
