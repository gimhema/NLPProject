use std::collections::HashMap;
use std::hash::Hash;
use std::f32;

/// 유저 고민/소원 카테고리
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Capital, // 자본/재화
    Growth,  // 육성/레벨/스펙
    Craft,   // 제작/강화/재련
    Party,   // 파티/협동
    Misc,    // 기타
}

/// 규칙 기반 키워드 하나
#[derive(Debug, Clone, Copy)]
pub struct KeywordRule {
    pub pattern: &'static str,
    pub weight: i32,
}

/// "레이블 + 키워드 규칙들" 한 묶음.
/// T는 Category 뿐 아니라, 나중에 감정(Sentiment) 같은 다른 enum도 재사용 가능.
#[derive(Debug, Clone, Copy)]
pub struct Rule<T> {
    pub label: T,
    pub keywords: &'static [KeywordRule],
}

/// 규칙 기반 점수 계산 (키워드 매칭)
///
/// threshold 이상인 것만 결과에 포함.
pub fn classify_with_rules<T>(text: &str, rules: &[Rule<T>], threshold: i32) -> Vec<(T, i32)>
where
    T: Copy + Eq + Hash,
{
    let normalized = normalize(text);

    let mut scores: HashMap<T, i32> = HashMap::new();

    for rule in rules {
        let mut rule_score: i32 = 0;
        for kw in rule.keywords {
            if normalized.contains(kw.pattern) {
                rule_score += kw.weight;
            }
        }

        if rule_score > 0 {
            let entry = scores.entry(rule.label).or_insert(0);
            *entry += rule_score;
        }
    }

    let mut result: Vec<(T, i32)> = scores
        .into_iter()
        .filter(|&(_, score)| score >= threshold)
        .collect();

    // 점수 내림차순 정렬
    result.sort_by(|a, b| b.1.cmp(&a.1));

    result
}

/// 간단한 소문자화 + 공백 정리
pub fn normalize(text: &str) -> String {
    let lower = text.to_lowercase();
    let trimmed = lower.trim();
    // 아주 단순하게 연속 공백만 하나로 축소
    let mut result = String::with_capacity(trimmed.len());
    let mut prev_space = false;
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                result.push(' ');
                prev_space = true;
            }
        } else {
            prev_space = false;
            result.push(ch);
        }
    }
    result
}

/// 학습에 사용할 라벨+텍스트 샘플
///  - Category: 자본/육성/제작/파티/기타
///  - text: 실제 채팅 내용
#[derive(Debug, Clone)]
pub struct LabeledSample {
    pub category: Category,
    pub text: String,
}

/// 텍스트를 TF-IDF 벡터로 바꾸는 벡터라이저
///
/// - vocabulary: 토큰 목록
/// - idf: 각 토큰의 IDF 값
/// - token_to_index: 토큰 → 인덱스 매핑
#[derive(Debug, Clone)]
pub struct TextVectorizer {
    pub vocabulary: Vec<String>,
    pub idf: Vec<f32>,
    pub token_to_index: HashMap<String, usize>,
}

impl TextVectorizer {
    /// 학습 샘플 전체를 보고 vocabulary + IDF 계산
    pub fn fit(samples: &[LabeledSample]) -> TextVectorizer {
        let mut df: HashMap<String, usize> = HashMap::new(); // document frequency
        let mut vocabulary: Vec<String> = Vec::new();
        let mut token_to_index: HashMap<String, usize> = HashMap::new();

        let n_docs = samples.len().max(1);

        for sample in samples {
            let normalized = normalize(&sample.text);
            let tokens = tokenize(&normalized);

            // 한 문서 안에서 중복 토큰은 df에 한 번만 카운트
            let mut seen_in_doc: HashMap<String, bool> = HashMap::new();
            for tok in tokens {
                if tok.is_empty() {
                    continue;
                }
                if !seen_in_doc.contains_key(&tok) {
                    seen_in_doc.insert(tok.clone(), true);
                    let counter = df.entry(tok.clone()).or_insert(0);
                    *counter += 1;
                    if !token_to_index.contains_key(&tok) {
                        let idx = vocabulary.len();
                        vocabulary.push(tok.clone());
                        token_to_index.insert(tok, idx);
                    }
                }
            }
        }

        // IDF 계산: log( N / (df + 1) )
        let mut idf: Vec<f32> = Vec::with_capacity(vocabulary.len());
        for tok in &vocabulary {
            let df_count = *df.get(tok).unwrap_or(&0) as f32;
            let val = (n_docs as f32 / (df_count + 1.0)).ln();
            idf.push(val);
        }

        TextVectorizer {
            vocabulary,
            idf,
            token_to_index,
        }
    }

    /// 하나의 텍스트를 TF-IDF 벡터로 변환
    pub fn vectorize(&self, text: &str) -> Vec<f32> {
        let normalized = normalize(text);
        let tokens = tokenize(&normalized);

        let n = self.vocabulary.len();
        let mut tf: Vec<f32> = vec![0.0; n];

        if tokens.is_empty() {
            return tf;
        }

        // term frequency (단순 count)
        for tok in tokens {
            if let Some(&idx) = self.token_to_index.get(&tok) {
                tf[idx] += 1.0;
            }
        }

        // TF-IDF: tf * idf, 그리고 L2 normalize
        let mut vec: Vec<f32> = vec![0.0; n];
        for i in 0..n {
            vec[i] = tf[i] * self.idf[i];
        }

        l2_normalize(&mut vec);
        vec
    }
}

/// 간단한 공백 기반 토큰화.
/// 필요하면 나중에 한국어 맞게 조정 가능.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// v를 L2 노말라이즈 (길이 1로)
fn l2_normalize(v: &mut [f32]) {
    let mut sum = 0.0;
    for &x in v.iter() {
        sum += x * x;
    }
    if sum <= 0.0 {
        return;
    }
    let norm = sum.sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// 카테고리별 k-means 클러스터링 결과.
///
/// 각 카테고리마다 여러 centroid 벡터를 보유.
#[derive(Debug, Clone)]
pub struct KMeansRefiner {
    pub centroids: HashMap<Category, Vec<Vec<f32>>>,
}

impl KMeansRefiner {
    pub fn new() -> KMeansRefiner {
        KMeansRefiner {
            centroids: HashMap::new(),
        }
    }
}

/// 카테고리별로 따로 k-means 학습을 수행.
/// - samples: 라벨(카테고리) + 텍스트
/// - vectorizer: 미리 fit 된 TextVectorizer
/// - k_per_category: 카테고리당 centroid 개수
/// - max_iter: k-means 반복 횟수
pub fn train_kmeans_by_category(
    samples: &[LabeledSample],
    vectorizer: &TextVectorizer,
    k_per_category: usize,
    max_iter: usize,
) -> KMeansRefiner {
    let mut refiner = KMeansRefiner::new();

    // 카테고리별로 벡터 모으기
    let mut per_cat: HashMap<Category, Vec<Vec<f32>>> = HashMap::new();

    for sample in samples {
        let v = vectorizer.vectorize(&sample.text);
        per_cat.entry(sample.category).or_insert(Vec::new()).push(v);
    }

    for (cat, vecs) in per_cat.into_iter() {
        if vecs.is_empty() {
            continue;
        }

        let k = k_per_category.min(vecs.len()).max(1);
        let centroids = kmeans_train(&vecs, k, max_iter);
        refiner.centroids.insert(cat, centroids);
    }

    refiner
}

/// 단일 데이터셋에 대한 k-means 학습 (유클리드 거리, L2노말라이즈 가정)
fn kmeans_train(vectors: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    let n = vectors.len();
    if n == 0 {
        return Vec::new();
    }

    let dim = vectors[0].len();
    let mut centroids: Vec<Vec<f32>> = Vec::new();

    // 초기 centroid: 앞에서 k개를 그대로 사용 (랜덤 사용 안 함: 외부 크레이트 X)
    for i in 0..k {
        centroids.push(vectors[i].clone());
    }

    let mut assignments: Vec<usize> = vec![0; n];

    for _ in 0..max_iter {
        // 할당 단계
        let mut changed = false;
        for (i, vec) in vectors.iter().enumerate() {
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;

            for (c_idx, c) in centroids.iter().enumerate() {
                let d = euclidean_distance_squared(vec, c);
                if d < best_dist {
                    best_dist = d;
                    best_idx = c_idx;
                }
            }

            if assignments[i] != best_idx {
                assignments[i] = best_idx;
                changed = true;
            }
        }

        if !changed {
            break; // 수렴
        }

        // 업데이트 단계
        let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts: Vec<usize> = vec![0; k];

        for (vec, &cluster_idx) in vectors.iter().zip(assignments.iter()) {
            let cnt = &mut counts[cluster_idx];
            *cnt += 1;
            let centroid = &mut new_centroids[cluster_idx];
            for d in 0..dim {
                centroid[d] += vec[d];
            }
        }

        for c_idx in 0..k {
            if counts[c_idx] == 0 {
                // 빈 클러스터면 기존 centroid 유지
                new_centroids[c_idx] = centroids[c_idx].clone();
            } else {
                let inv = 1.0 / (counts[c_idx] as f32);
                for d in 0..dim {
                    new_centroids[c_idx][d] *= inv;
                }
                // 중심도 L2 normalize
                l2_normalize(&mut new_centroids[c_idx]);
            }
        }

        centroids = new_centroids;
    }

    centroids
}

fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let len = a.len().min(b.len());
    for i in 0..len {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// cosine similarity 계산
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    let len = a.len().min(b.len());
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// 규칙 기반 점수를 k-means centroid와 비교해서 보정
///
/// - rule_scores: 규칙 기반 (Category, 점수)
/// - 반환: 보정된 점수 (f32)
pub fn refine_category_scores_with_kmeans(
    text: &str,
    rule_scores: &[(Category, i32)],
    vectorizer: &TextVectorizer,
    refiner: &KMeansRefiner,
) -> Vec<(Category, f32)> {
    let vec = vectorizer.vectorize(text);
    let mut results: Vec<(Category, f32)> = Vec::new();

    for &(cat, score) in rule_scores.iter() {
        let mut final_score = score as f32;
        if let Some(centroids) = refiner.centroids.get(&cat) {
            let mut max_sim = 0.0_f32;
            for c in centroids.iter() {
                let sim = cosine_similarity(&vec, &c);
                if sim > max_sim {
                    max_sim = sim;
                }
            }

            // 단순한 보정 규칙 예시:
            // - 유사도가 너무 낮으면 패널티
            // - 유사도가 높으면 보너스
            if max_sim < 0.1 {
                final_score *= 0.5;
            } else if max_sim > 0.6 {
                final_score *= 1.2;
            }
        }

        results.push((cat, final_score));
    }

    // 점수 내림차순 정렬
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}
