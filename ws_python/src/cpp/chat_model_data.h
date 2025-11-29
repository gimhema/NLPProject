// chat_model_data.h
// 오프라인 학습 결과를 C++에서 바로 사용할 수 있게 만든 정적 데이터
#pragma once

#include <string>

// 실제 모델 크기 (Python 쪽에서 export할 때 채우기)
static const int CHAT_NUM_CLASSES  = 3;   // 예: Capital, Battle, Pet
static const int CHAT_NUM_FEATURES = 128; // 예시. 실제 TF-IDF vocab 크기로 변경

// 카테고리 이름
static const char* CHAT_CATEGORY_NAMES[CHAT_NUM_CLASSES] = {
    "Capital",
    "Battle",
    "Pet"
    // 필요시 더 추가
};

// vocab: 토큰 문자열과 그 토큰에 대응되는 인덱스
// 실제로는 CHAT_NUM_FEATURES 개 만큼 채워야 함
struct ChatTokenEntry {
    const char* token;
    int index;
};

// 예시: 실제로는 Python에서 vocab(dict: token->index)을 그대로 펼쳐서 만들어야 함
static const ChatTokenEntry CHAT_VOCAB_ENTRIES[CHAT_NUM_FEATURES] = {
    // { "나는", 0 },
    // { "빌게이츠처럼", 1 },
    // ...
    // Python에서 vocabulary_ 순서대로 채우면 됨
};

// idf 벡터 (size = CHAT_NUM_FEATURES)
static const float CHAT_IDF[CHAT_NUM_FEATURES] = {
    // 예: 3.1f, 2.7f, ...
};

// 로지스틱 회귀 or Linear SVM weight (coef)
// shape: [CHAT_NUM_CLASSES][CHAT_NUM_FEATURES]
static const float CHAT_WEIGHTS[CHAT_NUM_CLASSES][CHAT_NUM_FEATURES] = {
    // 예시:
    // { 0.5f, 0.7f, 1.2f, ... }, // Capital
    // { -0.1f, -0.3f, 1.5f, ... }, // Battle
    // { ... } // Pet
};

// bias (intercept) shape: [CHAT_NUM_CLASSES]
static const float CHAT_BIAS[CHAT_NUM_CLASSES] = {
    // 예: 0.0f, 0.1f, -0.05f
};

