from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


from ko_tokenizer import tokenize_ko


@dataclass
class TfidfConfig:
    ngram_min: int = 1
    ngram_max: int = 1
    min_df: int = 1
    max_df: float = 1.0
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = False

    # (옵션) 필요하면 향후 확장용으로 keep_tags 같은 걸 config에 넣어도 됨


class TfidfTrainer:
    def __init__(self, config: TfidfConfig | None = None):
        self.config = config or TfidfConfig()
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self.inv_vocab: Dict[int, str] = {}

        self.vectorizer = self._build_vectorizer(self.config)

    def _build_vectorizer(self, cfg: TfidfConfig) -> TfidfVectorizer:
        return TfidfVectorizer(
            ngram_range=(cfg.ngram_min, cfg.ngram_max),
            min_df=cfg.min_df,
            max_df=cfg.max_df,
            use_idf=cfg.use_idf,
            smooth_idf=cfg.smooth_idf,
            sublinear_tf=cfg.sublinear_tf,

            tokenizer=tokenize_ko,
            token_pattern=None,   # 중요: tokenizer를 쓰는 경우 None
            lowercase=False,
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
        idf = self.vectorizer.idf_.tolist() if hasattr(self.vectorizer, "idf_") else []

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
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = data.get("config", {})

        # 안전하게 기본값 사용
        ngram_min = cfg.get("ngram_min", self.config.ngram_min)
        ngram_max = cfg.get("ngram_max", self.config.ngram_max)
        min_df = cfg.get("min_df", self.config.min_df)
        max_df = cfg.get("max_df", self.config.max_df)
        use_idf = cfg.get("use_idf", self.config.use_idf)
        smooth_idf = cfg.get("smooth_idf", self.config.smooth_idf)
        sublinear_tf = cfg.get("sublinear_tf", self.config.sublinear_tf)

        vocab = data.get("vocab", {})
        idf = data.get("idf", [])

        # 내부 상태 갱신
        self.config.ngram_min = ngram_min
        self.config.ngram_max = ngram_max
        self.config.min_df = min_df
        self.config.max_df = max_df
        self.config.use_idf = use_idf
        self.config.smooth_idf = smooth_idf
        self.config.sublinear_tf = sublinear_tf

        self.vocab = vocab
        self.idf = idf

        # ✅ 형태소 토크나이저 설정 포함해서 재생성 (가장 중요)
        self.vectorizer = self._build_vectorizer(self.config)

        # vocabulary_ 와 idf_ 를 직접 주입
        try:
            self.vectorizer.vocabulary_ = {k: int(v) for k, v in vocab.items()}
        except Exception:
            self.vectorizer.vocabulary_ = vocab

        if idf:
            self.vectorizer.idf_ = np.asarray(idf, dtype=float)
        else:
            self.vectorizer.idf_ = np.array([])

        self.inv_vocab = {
            int(idx): term for term, idx in self.vectorizer.vocabulary_.items()
        }
