
use std::io;

use crate::data_loader;
use crate::offline_algo::{
    Category, KeywordRule, Rule, LabeledSample,
    TextVectorizer, KMeansRefiner,
    classify_with_rules, refine_category_scores_with_kmeans,
};

// ====== 규칙 테이블 정의 ======

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
    KeywordRule { pattern: "파티",     weight: 2 },
    KeywordRule { pattern: "구인",     weight: 2 },
    KeywordRule { pattern: "구함",     weight: 2 },
    KeywordRule { pattern: "같이 가실", weight: 3 },
    KeywordRule { pattern: "던전 같이", weight: 3 },
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
            rule_threshold: 1, // 기본 threshold
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
            // 학습 데이터가 없으면 그냥 규칙 기반 모드만 사용
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
        // 1단계: 규칙 기반으로 1차 필터
        let mut rule_scores =
            classify_with_rules::<Category>(text, CATEGORY_RULES, self.rule_threshold);

        if rule_scores.is_empty() {
            // 어느 카테고리에도 크게 속하지 않으면 Misc 처리
            return vec![(Category::Misc, 0.0)];
        }

        // 2단계: k-means로 보정
        match (&self.vectorizer, &self.kmeans) {
            (Some(vec), Some(kmeans)) => {
                refine_category_scores_with_kmeans(text, &rule_scores, vec, kmeans)
            }
            _ => {
                // 학습이 안 되어 있으면 규칙 점수만 f32로 변환해서 사용
                let mut res: Vec<(Category, f32)> = Vec::new();
                for (cat, s) in rule_scores.drain(..) {
                    res.push((cat, s as f32));
                }
                res
            }
        }
    }
}
