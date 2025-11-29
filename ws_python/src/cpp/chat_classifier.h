// chat_classifier.h
#pragma once

#include <string>
#include <map>
#include <vector>

class ChatClassifier
{
public:
    ChatClassifier();

    // 정적 모델(chat_model_data.h)을 기반으로 내부 자료구조 초기화
    // 프로그램 시작 시 1번만 호출하면 됨.
    bool initialize();

    // 입력 채팅을 카테고리 이름 문자열("Capital" 등)로 분류
    // outConfidence가 NULL이 아니면 softmax 기반의 확률 유사 값을 [0,1] 범위로 돌려줌
    std::string classify(const std::string& text, float* outConfidence /*= NULL*/) const;

    // argmax로 뽑은 클래스의 인덱스를 반환 (0 ~ num_classes-1)
    int classifyIndex(const std::string& text, float* outConfidence /*= NULL*/) const;

private:
    // 토큰 → 인덱스 (TF-IDF feature index)
    std::map<std::string, int> m_vocab;
    int m_numClasses;
    int m_numFeatures;

private:
    void buildVocabFromStaticData();
    void tokenize(const std::string& text, std::vector<std::string>& outTokens) const;
    void computeScores(const std::vector<std::string>& tokens,
                       std::vector<float>& outScores) const;
};
