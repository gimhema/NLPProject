// chat_classifier.cpp
#include "chat_classifier.h"
#include "chat_model_data.h"

#include <cctype>   // std::isspace, std::ispunct
#include <cmath>    // std::exp
#include <iostream> // 디버깅용 (원치 않으면 제거 가능)

ChatClassifier::ChatClassifier()
    : m_numClasses(0)
    , m_numFeatures(0)
{
}

bool ChatClassifier::initialize()
{
    // chat_model_data.h 에 정의된 상수를 가져와서 세팅
    m_numClasses  = CHAT_NUM_CLASSES;
    m_numFeatures = CHAT_NUM_FEATURES;

    if (m_numClasses <= 0 || m_numFeatures <= 0) {
        return false;
    }

    buildVocabFromStaticData();
    return true;
}

void ChatClassifier::buildVocabFromStaticData()
{
    m_vocab.clear();

    int i;
    for (i = 0; i < CHAT_NUM_FEATURES; ++i) {
        const ChatTokenEntry& e = CHAT_VOCAB_ENTRIES[i];
        if (e.token != 0 && e.index >= 0 && e.index < CHAT_NUM_FEATURES) {
            // token 문자열 → index 매핑
            m_vocab[std::string(e.token)] = e.index;
        }
    }
}

void ChatClassifier::tokenize(const std::string& text, std::vector<std::string>& outTokens) const
{
    // 매우 단순한 토크나이저:
    // - 공백/구두점 기준으로 split
    // - UTF-8 한글의 경우, 공백 단위로 쪼개진다고 생각하고 동작 (세부 형태소 분석은 안함)

    outTokens.clear();
    std::string current;

    std::string::size_type i;
    for (i = 0; i < text.size(); ++i) {
        unsigned char ch = static_cast<unsigned char>(text[i]);

        // 공백 또는 ASCII 구두점 기준으로 토큰 나누기
        if (std::isspace(ch) || std::ispunct(ch)) {
            if (!current.empty()) {
                outTokens.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(text[i]);
        }
    }

    if (!current.empty()) {
        outTokens.push_back(current);
    }
}

void ChatClassifier::computeScores(const std::vector<std::string>& tokens,
                                   std::vector<float>& outScores) const
{
    outScores.assign(m_numClasses, 0.0f);

    if (tokens.empty()) {
        // 토큰이 없으면 그냥 bias만 사용
        int c;
        for (c = 0; c < m_numClasses; ++c) {
            outScores[c] = CHAT_BIAS[c];
        }
        return;
    }

    // 1) term frequency (index → count)
    std::map<int, int> termFreq;
    int totalTokens = 0;

    int t;
    for (t = 0; t < (int)tokens.size(); ++t) {
        std::map<std::string,int>::const_iterator it = m_vocab.find(tokens[t]);
        if (it == m_vocab.end()) {
            continue; // vocab에 없는 토큰은 무시
        }
        int idx = it->second;
        std::map<int,int>::iterator fit = termFreq.find(idx);
        if (fit == termFreq.end()) {
            termFreq[idx] = 1;
        } else {
            fit->second += 1;
        }
        totalTokens += 1;
    }

    if (totalTokens == 0) {
        // 전부 unknown token 이라면 bias만 사용
        int c;
        for (c = 0; c < m_numClasses; ++c) {
            outScores[c] = CHAT_BIAS[c];
        }
        return;
    }

    // 2) 각 클래스별로 W·x + b 계산
    int c;
    for (c = 0; c < m_numClasses; ++c) {
        float score = CHAT_BIAS[c];

        std::map<int,int>::const_iterator it = termFreq.begin();
        for (; it != termFreq.end(); ++it) {
            int idx  = it->first;
            int cnt  = it->second;

            if (idx < 0 || idx >= m_numFeatures) {
                continue;
            }

            // TF
            float tf = (float)cnt / (float)totalTokens;
            // IDF
            float idf = CHAT_IDF[idx];
            // TF-IDF
            float tfidf = tf * idf;

            score += CHAT_WEIGHTS[c][idx] * tfidf;
        }

        outScores[c] = score;
    }
}

int ChatClassifier::classifyIndex(const std::string& text, float* outConfidence) const
{
    std::vector<std::string> tokens;
    tokenize(text, tokens);

    std::vector<float> scores;
    computeScores(tokens, scores);

    if (scores.empty()) {
        if (outConfidence) {
            *outConfidence = 0.0f;
        }
        return -1;
    }

    // argmax
    int bestIndex = 0;
    float bestScore = scores[0];

    int c;
    for (c = 1; c < m_numClasses; ++c) {
        if (scores[c] > bestScore) {
            bestScore = scores[c];
            bestIndex = c;
        }
    }

    // softmax 기반 confidence 계산 (선택 사항)
    if (outConfidence) {
        double sumExp = 0.0;
        for (c = 0; c < m_numClasses; ++c) {
            sumExp += std::exp((double)scores[c]);
        }
        double prob = 0.0;
        if (sumExp > 0.0) {
            prob = std::exp((double)scores[bestIndex]) / sumExp;
        }
        *outConfidence = (float)prob;
    }

    return bestIndex;
}

std::string ChatClassifier::classify(const std::string& text, float* outConfidence) const
{
    int idx = classifyIndex(text, outConfidence);
    if (idx < 0 || idx >= m_numClasses) {
        return std::string(); // 빈 문자열 = Unknown
    }

    return std::string(CHAT_CATEGORY_NAMES[idx]);
}
