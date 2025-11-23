// offline_emotion.rs

use std::io;

use crate::data_loader;
use crate::offline_algo::{
    Category, KeywordRule, Rule, LabeledSample,
    TextVectorizer, KMeansRefiner,
    classify_with_rules, refine_category_scores_with_kmeans,
};

/// ====== 카테고리용 규칙 테이블 ======

const CAPITAL_KEYWORDS: &[KeywordRule] = &[
    KeywordRule { pattern: "골드",   weight: 2 },
    KeywordRule { pattern: "돈",     weight: 2 },
    KeywordRule { pattern: "재화",   weight: 2 },
    KeywordRule { pattern: "거지",   weight: 3 },
    KeywordRule { pattern: "파산",   weight: 3 },
    KeywordRule { pattern: "없어",   weight: 1 },
    KeywordRule { pattern: "모자라", weight: 1 },
];

const GROWTH_KEYWORDS: &[KeywordRule] = &[
    KeywordRule { pattern: "레벨",   weight: 2 },
    KeywordRule { pattern: "렙",     weight: 2 },
    KeywordRule { pattern: "경험치", weight: 2 },
    KeywordRule { pattern: "육성",   weight: 2 },
    KeywordRule { pattern: "스펙업", weight: 3 },
    KeywordRule { pattern: "스탯",   weight: 2 },
];

const CRAFT_KEYWORDS: &[KeywordRule] = &[
    KeywordRule { pattern: "강화 실패", weight: 4 }, // 프레이즈 우선
    KeywordRule { pattern: "강화",      weight: 3 },
    KeywordRule { pattern: "재련",      weight: 3 },
    KeywordRule { pattern: "제작",      weight: 2 },
    KeywordRule { pattern: "합성",      weight: 2 },
    KeywordRule { pattern: "업그레이드",weight: 2 },
];

const PARTY_KEYWORDS: &[KeywordRule] = &[
    KeywordRule { pattern: "파티",      weight: 2 },
    KeywordRule { pattern: "구인",      weight: 2 },
    KeywordRule { pattern: "구함",      weight: 2 },
    KeywordRule { pattern: "같이 가실",  weight: 3 },
    KeywordRule { pattern: "던전 같이",  weight: 3 },
];

const CAPITAL_RULE: Rule<Category> = Rule {
    label: Category::Capital,
    keywords: CAPITAL_KEYWORDS,
};
const GROWTH_RULE: Rule<Category> = Rule {
    label: Category::Growth,
    keywords: GROWTH_KEYWORDS,
};
const CRAFT_RULE: Rule<Category> = Rule {
    label: Category::Craft,
    keywords: CRAFT_KEYWORDS,
};
const PARTY_RULE: Rule<Category> = Rule {
    label: Category::Party,
    keywords: PARTY_KEYWORDS,
};

const CATEGORY_RULES: &[Rule<Category>] = &[
    CAPITAL_RULE,
    GROWTH_RULE,
    CRAFT_RULE,
    PARTY_RULE,
];

/// 오프라인 카테고리 분석 엔진.
/// - 규칙 기반 분류 + (선택) k-means 보정
pub struct OfflineCategoryEngine {
    pub vectorizer: Option<TextVectorizer>,
    pub kmeans: Option<KMeansRefiner>,
    /// 규칙 기반 점수 threshold
    pub rule_threshold: i32,
}

impl OfflineCategoryEngine {
    pub fn new() -> OfflineCategoryEngine {
        OfflineCategoryEngine {
            vectorizer: None,
            kmeans: None,
            rule_threshold: 1,
        }
    }

    /// 학습 데이터를 로드하고,
    /// - TextVectorizer 학습
    /// - 카테고리별 k-means 학습
    pub fn train_from_tsv(
        &mut self,
        path: &str,
        k_per_category: usize,
        max_iter: usize,
    ) -> io::Result<()> {
        let samples: Vec<LabeledSample> = data_loader::load_labeled_samples_from_tsv(path)?;
        if samples.is_empty() {
            // 학습 데이터가 없으면 k-means 없이 규칙 기반만 사용
            return Ok(());
        }

        let vectorizer = TextVectorizer::fit(&samples);
        let kmeans = crate::offline_algo::train_kmeans_by_category(
            &samples,
            &vectorizer,
            k_per_category,
            max_iter,
        );

        self.vectorizer = Some(vectorizer);
        self.kmeans = Some(kmeans);

        Ok(())
    }

    /// 유저 채팅 한 줄을 분석해서
    /// - 규칙 기반 + k-means 보정 점수 반환
    ///
    /// 반환값: (Category, score) 높은 순으로 정렬
    pub fn analyze(&self, text: &str) -> Vec<(Category, f32)> {
        // 1단계: 규칙 기반
        let mut rule_scores =
            classify_with_rules::<Category>(text, CATEGORY_RULES, self.rule_threshold);

        if rule_scores.is_empty() {
            // 어느 카테고리에도 크게 속하지 않으면 Misc 처리
            return vec![(Category::Misc, 0.0)];
        }

        // 2단계: k-means 보정 (있을 때만)
        match (&self.vectorizer, &self.kmeans) {
            (Some(vec), Some(kmeans)) => {
                refine_category_scores_with_kmeans(text, &rule_scores, vec, kmeans)
            }
            _ => {
                // 학습 없으면 규칙 점수만 사용
                let mut res: Vec<(Category, f32)> = Vec::new();
                for (cat, s) in rule_scores.drain(..) {
                    res.push((cat, s as f32));
                }
                res
            }
        }
    }
}

/// ====== 감정(Sentiment)용 정의 ======

use std::collections::HashMap;
use std::hash::Hash;

/// 감정 태그
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sentiment {
    Positive,
    Negative,
    Sad,
    Angry,
    Neutral,
}

// 감정 키워드 규칙

const POSITIVE_KEYWORDS: &[KeywordRule] = &[
    KeywordRule { pattern: "기분 좋", weight: 3 },
    KeywordRule { pattern: "좋다",   weight: 2 },
    KeywordRule { pattern: "행복",   weight: 3 },
    KeywordRule { pattern: "재밌",   weight: 2 },
    KeywordRule { pattern: "즐겁",   weight: 2 },
    KeywordRule { pattern: "운 좋",  weight: 2 },
];

const NEGATIVE_KEYWORDS: &[KeywordRule] = &[
    KeywordRule { pattern: "별로",   weight: 1 },
    KeywordRule { pattern: "싫다",   weight: 2 },
    KeywordRule { pattern: "안 좋아", weight: 2 },
    KeywordRule { pattern: "최악",   weight: 3 },
];

const SAD_KEYWORDS: &[KeywordRule] = &[
    KeywordRule { pattern: "슬프",   weight: 3 },
    KeywordRule { pattern: "우울",   weight: 3 },
    KeywordRule { pattern: "힘들어", weight: 2 },
    KeywordRule { pattern: "눈물",   weight: 2 },
    KeywordRule { pattern: "외롭",   weight: 2 },
    KeywordRule { pattern: "포기하고 싶", weight: 4 },
];

const ANGRY_KEYWORDS: &[KeywordRule] = &[
    KeywordRule { pattern: "빡치",   weight: 3 },
    KeywordRule { pattern: "짜증",   weight: 2 },
    KeywordRule { pattern: "화가 나", weight: 3 },
    KeywordRule { pattern: "열받",   weight: 3 },
    KeywordRule { pattern: "개같",   weight: 2 },
    KeywordRule { pattern: "버그",   weight: 1 }, // 상황에 따라 가중치 조절
];

const NEUTRAL_KEYWORDS: &[KeywordRule] = &[
    KeywordRule { pattern: "그냥",   weight: 1 },
    KeywordRule { pattern: "그럭저럭", weight: 2 },
];

const POSITIVE_RULE: Rule<Sentiment> = Rule {
    label: Sentiment::Positive,
    keywords: POSITIVE_KEYWORDS,
};
const NEGATIVE_RULE: Rule<Sentiment> = Rule {
    label: Sentiment::Negative,
    keywords: NEGATIVE_KEYWORDS,
};
const SAD_RULE: Rule<Sentiment> = Rule {
    label: Sentiment::Sad,
    keywords: SAD_KEYWORDS,
};
const ANGRY_RULE: Rule<Sentiment> = Rule {
    label: Sentiment::Angry,
    keywords: ANGRY_KEYWORDS,
};
const NEUTRAL_RULE: Rule<Sentiment> = Rule {
    label: Sentiment::Neutral,
    keywords: NEUTRAL_KEYWORDS,
};

const SENTIMENT_RULES: &[Rule<Sentiment>] = &[
    POSITIVE_RULE,
    NEGATIVE_RULE,
    SAD_RULE,
    ANGRY_RULE,
    NEUTRAL_RULE,
];

/// 감정 분석 엔진 (규칙 기반)
pub struct OfflineSentimentEngine {
    pub rule_threshold: i32,
}

impl OfflineSentimentEngine {
    pub fn new() -> OfflineSentimentEngine {
        OfflineSentimentEngine {
            rule_threshold: 1,
        }
    }

    /// 유저 채팅 한 줄에 대해 감정 태그와 점수를 반환.
    /// - 보통 가장 점수가 높은 1개 또는 2개만 사용하면 됨.
    pub fn analyze(&self, text: &str) -> Vec<(Sentiment, i32)> {
        let mut result =
            classify_with_rules::<Sentiment>(text, SENTIMENT_RULES, self.rule_threshold);

        if result.is_empty() {
            // 아무 감정 키워드도 안 잡히면 Neutral 하나만 기본값으로
            result.push((Sentiment::Neutral, 0));
        }

        result
    }

    /// 가장 점수가 높은 감정 하나만 뽑고 싶을 때
    pub fn top_sentiment(&self, text: &str) -> Sentiment {
        let mut scored = self.analyze(text);
        if scored.is_empty() {
            return Sentiment::Neutral;
        }
        // 이미 classify_with_rules 안에서 점수 내림차순 정렬되어 있음
        scored[0].0
    }
}
