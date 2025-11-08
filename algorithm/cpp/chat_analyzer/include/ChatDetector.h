
#ifndef CHAT_DETECTOR_H_
#define CHAT_DETECTOR_H_

#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <cctype>
#include <algorithm>

// ====== (선택) unordered_map 지원: VS2010 TR1, GCC/Clang unordered_map ======
#if defined(_MSC_VER)
  // VS2010: TR1
  #include <functional>
  #include <tr1/unordered_map>
  namespace lag_unordered_ns = std::tr1;
#elif defined(__GNUC__) || defined(__clang__)
  #if __cplusplus >= 201103L
    #include <unordered_map>
    namespace lag_unordered_ns = std;
  #else
    // GNU++98에서도 구현체에 따라 <tr1/unordered_map> 가능
    #include <tr1/unordered_map>
    namespace lag_unordered_ns = std::tr1;
  #endif
#endif

namespace LagComplaint {

struct DetectResult {
    bool  is_complaint;
    float weight;
    int   hits;
};

struct Lexicon {
    std::vector<std::string> positives;
    std::vector<std::string> negations;
    std::vector<std::string> intensifiers;
    std::vector<std::string> interrogatives;
};

inline static std::string ToLowerAscii(const std::string& s) {
    std::string r(s);
    for (size_t i = 0; i < r.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(r[i]);
        if (c < 128) r[i] = static_cast<char>(std::tolower(c));
    }
    return r;
}

// 공백/구두점 기준 토큰화 (오버헤드 낮음)
inline static void Tokenize(const std::string& s,
                            std::vector<std::string>& tokens,
                            std::vector<size_t>& offsets) {
    tokens.clear(); offsets.clear();
    size_t i = 0, n = s.size();
    while (i < n) {
        while (i < n) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c <= 32 || c==',' || c=='.' || c=='!' || c=='?' || c==';' || c==':' ||
                c=='(' || c==')' || c=='[' || c==']' || c=='{' || c=='}' || c=='\"' || c=='\'')
                ++i;
            else break;
        }
        if (i >= n) break;
        size_t start = i;
        while (i < n) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c <= 32 || c==',' || c=='.' || c=='!' || c=='?' || c==';' || c==':' ||
                c=='(' || c==')' || c=='[' || c==']' || c=='{' || c=='}' || c=='\"' || c=='\'')
                break;
            ++i;
        }
        size_t end = i;
        if (end > start) {
            tokens.push_back(s.substr(start, end - start));
            offsets.push_back(start);
        }
    }
}

inline static bool ContainsSubstringFrom(const std::string& text,
                                         const std::string& needle,
                                         size_t& posInOut) {
    size_t p = text.find(needle, posInOut);
    if (p == std::string::npos) return false;
    posInOut = p;
    return true;
}

inline static int CharPosToTokenIndex(size_t charPos,
                                      const std::vector<std::string>& tokens,
                                      const std::vector<size_t>& offsets) {
    if (tokens.empty()) return -1;
    for (size_t i = 0; i < tokens.size(); ++i) {
        size_t s = offsets[i];
        size_t e = s + tokens[i].size();
        if (charPos >= s && charPos < e) return static_cast<int>(i);
    }
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (charPos < offsets[i]) return (i==0)? -1 : static_cast<int>(i-1);
    }
    return static_cast<int>(tokens.size()) - 1;
}

class Detector {
public:
    Detector()
    : window_radius_(4)
    , question_penalty_(0.6f)
    , intensify_bonus_(0.5f)
    , min_hit_weight_(1.0f)
    , cooldown_sec_(60)
    , ewma_lambda_(0.95)
    , ewma_(0.0)
    , last_update_sec_(0)
    , cusum_k_(0.2)
    , cusum_h_(3.0)
    , cusum_gpos_(0.0)
    , cusum_gneg_(0.0)
    , cusum_alarm_(false)
    , cleanup_interval_sec_(60)
    , last_cleanup_sec_(0)
    {
        LoadDefaultLexicon(klex_);
        BuildPositiveBuckets(); // (B) 버킷 구성
    }

    void SetCooldownSeconds(int s) { if (s>=0) cooldown_sec_ = s; }
    void SetWindowRadius(int r)    { if (r>=0) window_radius_ = r; }
    void SetEwmaLambda(double l)   { if (l>0.0 && l<1.0) ewma_lambda_ = l; }
    void SetCusum(double k, double h){ if (k>0) cusum_k_=k; if (h>0) cusum_h_=h; }
    void SetLexicon(const Lexicon& lx) { klex_ = lx; BuildPositiveBuckets(); }

    // 메시지 1건 분류
    DetectResult Classify(const std::string& rawText,
                          const std::string& userId,
                          std::time_t nowEpochSec)
    {
        DetectResult R; R.is_complaint=false; R.weight=0.0f; R.hits=0;

        // (C) 쿨다운 맵 청소 (드물게)
        MaybeCleanupCooldown(nowEpochSec);

        if (IsOnCooldown(userId, nowEpochSec)) return R;

        std::string text = ToLowerAscii(rawText);

        // (A) 조기 토큰화 회피: 먼저 빠르게 "긍정 키워드 존재?"만 검사
        if (!FastAnyPositiveHit(text)) {
            return R; // 일치 없음 → 바로 반환
        }

        // 여기까지 왔으면 매치 후보가 최소 1개 존재 → 토큰화/윈도우 판정
        std::vector<std::string> tokens; std::vector<size_t> offsets;
        tokens.reserve(32); offsets.reserve(32); // 동적할당 억제
        Tokenize(text, tokens, offsets);

        bool hasQuestion = (text.find('?') != std::string::npos) || ContainsAnyToken(tokens, klex_.interrogatives);

        float total = 0.0f;
        int hits = 0;

        // (B) 버킷을 사용해 첫 바이트 일치하는 키워드만 스캔
        for (size_t idxChar = 0; idxChar < text.size(); ++idxChar) {
            unsigned char lead = static_cast<unsigned char>(text[idxChar]);
            const std::vector<std::string>& bucket = pos_buckets_[lead];
            if (bucket.empty()) continue;

            // 버킷 내 후보들만 검사
            for (size_t k = 0; k < bucket.size(); ++k) {
                const std::string& key = bucket[k];
                // 현재 위치에서 key가 시작되는지 확인(짧은 memcmp 스타일)
                if (idxChar + key.size() <= text.size() &&
                    text.compare(idxChar, key.size(), key) == 0)
                {
                    // 토큰 윈도우 판정
                    int tidx = CharPosToTokenIndex(idxChar, tokens, offsets);
                    int L = (tidx < 0) ? 0 : (tidx - window_radius_);
                    int Rr= (tidx < 0) ? -1 : (tidx + window_radius_);
                    if (L < 0) L = 0;
                    if (Rr >= (int)tokens.size()) Rr = (int)tokens.size()-1;

                    bool neg = false, intens = false;
                    for (int t = L; t <= Rr && t >= 0; ++t) {
                        if (!neg   && IsInList(tokens[t], klex_.negations))   neg = true;
                        if (!intens&& IsInList(tokens[t], klex_.intensifiers)) intens = true;
                        if (neg && intens) break;
                    }

                    if (!neg) {
                        float w = min_hit_weight_;
                        if (intens) w += intensify_bonus_;
                        if (hasQuestion) w *= question_penalty_;
                        total += w;
                        ++hits;

                        // 임계 초과 시 조기 종료(선택)
                        if (total >= 1.0f && hits >= 2) {
                            // 충분히 확신할 때 빠르게 탈출하여 성능 절약
                            idxChar = text.size(); // 외부 루프 탈출
                            break;
                        }
                    }
                }
            }
        }

        R.hits = hits;
        R.is_complaint = (total >= 1.0f);
        R.weight = total;

        if (R.is_complaint && cooldown_sec_ > 0)
            cooldown_map_[userId] = nowEpochSec;

        return R;
    }

    // 시계열 지표 업데이트
    void UpdateMetrics(double value, std::time_t nowEpochSec) {
        if (last_update_sec_ == 0) last_update_sec_ = nowEpochSec;
        ewma_ = ewma_lambda_ * ewma_ + (1.0 - ewma_lambda_) * value;

        double xt = value;
        cusum_gpos_ = (cusum_gpos_ + (xt - cusum_k_));
        if (cusum_gpos_ < 0.0) cusum_gpos_ = 0.0;
        cusum_gneg_ = (cusum_gneg_ - (xt + cusum_k_));
        if (cusum_gneg_ < 0.0) cusum_gneg_ = 0.0;

        cusum_alarm_ = (cusum_gpos_ > cusum_h_);
        last_update_sec_ = nowEpochSec;
    }

    double EwmaScore() const { return ewma_; }
    bool   CusumAlarm() const { return cusum_alarm_; }

    // 디폴트 사전 로드
    static void LoadDefaultLexicon(Lexicon& lx) {
        const char* pos[] = {
            // 한글(UTF-8/ANSI와 무관, 바이트 서브스트링 매칭)
            "렉", "랙", "끊김", "끊기", "지연", "딜레이", "느리", "버벅", "핑 높", "핑튐",
            "프레임 드랍", "프레임떡락", "튕김", "멈춤", "밀림", "반응 늦", "서버 터짐",
            // 영문
            "lag", "delay", "stutter", "stuttering", "high ping", "ping spike", "frame drop", "freeze"
        };
        const char* neg[] = {
            "아님", "아니", "안", "없", "괜찮", "문제 아님", "착각", "내 인터넷", "와이파이", "옵션 때문",
            "not", "no", "dont", "doesnt", "fine", "ok"
        };
        const char* inten[] = {
            "엄청", "개", "너무", "진짜", "계속", "자꾸", "심함", "완전", "존나"
        };
        const char* interr[] = {
            "인가", "인가요", "인거", "인듯", "인 것 같", "인가?", "lag?", "ping?"
        };
        lx.positives.clear(); lx.negations.clear(); lx.intensifiers.clear(); lx.interrogatives.clear();
        for (size_t i=0;i<sizeof(pos)/sizeof(pos[0]);++i) lx.positives.push_back(ToLowerAscii(pos[i]));
        for (size_t i=0;i<sizeof(neg)/sizeof(neg[0]);++i) lx.negations.push_back(ToLowerAscii(neg[i]));
        for (size_t i=0;i<sizeof(inten)/sizeof(inten[0]);++i) lx.intensifiers.push_back(ToLowerAscii(inten[i]));
        for (size_t i=0;i<sizeof(interr)/sizeof(interr[0]);++i) lx.interrogatives.push_back(ToLowerAscii(interr[i]));
    }

private:
    // ===== 내부 상태 =====
    int   window_radius_;
    float question_penalty_;
    float intensify_bonus_;
    float min_hit_weight_;

    int cooldown_sec_;

    // (C) 쿨다운 테이블: 기본은 std::map, 가능시 unordered_map으로 교체
#if defined(lag_unordered_ns)
    lag_unordered_ns::unordered_map<std::string, std::time_t> cooldown_map_;
#else
    std::map<std::string, std::time_t> cooldown_map_;
#endif

    double      ewma_lambda_;
    double      ewma_;
    std::time_t last_update_sec_;

    // CUSUM
    double cusum_k_;
    double cusum_h_;
    double cusum_gpos_;
    double cusum_gneg_;
    bool   cusum_alarm_;

    Lexicon klex_;

    // (B) 첫 바이트 버킷
    std::vector<std::string> pos_buckets_[256];

    // (C) 청소 주기
    int         cleanup_interval_sec_;
    std::time_t last_cleanup_sec_;

private:
    // (A) 긍정 키워드 존재 여부 빠른 체크
    bool FastAnyPositiveHit(const std::string& text) const {
        // 첫 바이트 버킷을 사용해서 최소 1개라도 있는지 빠르게 확인
        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char lead = static_cast<unsigned char>(text[i]);
            const std::vector<std::string>& bucket = pos_buckets_[lead];
            if (bucket.empty()) continue;
            for (size_t k = 0; k < bucket.size(); ++k) {
                const std::string& key = bucket[k];
                if (i + key.size() <= text.size() &&
                    text.compare(i, key.size(), key) == 0) {
                    return true;
                }
            }
        }
        return false;
    }

    void BuildPositiveBuckets() {
        for (int b=0;b<256;++b) pos_buckets_[b].clear();
        for (size_t i=0;i<klex_.positives.size();++i) {
            const std::string s = ToLowerAscii(klex_.positives[i]);
            if (s.empty()) continue;
            unsigned char lead = static_cast<unsigned char>(s[0]);
            pos_buckets_[lead].push_back(s);
        }
        // 길이 긴 키 먼저 검사하면 실전에서 유리한 경우가 있어 정렬(선택)
        for (int b=0;b<256;++b) {
            std::vector<std::string>& v = pos_buckets_[b];
            std::sort(v.begin(), v.end(), LengthDescThenLex());
        }
    }

    struct LengthDescThenLex {
        bool operator()(const std::string& a, const std::string& b) const {
            if (a.size() != b.size()) return a.size() > b.size();
            return a < b;
        }
    };

    bool IsOnCooldown(const std::string& uid, std::time_t nowSec) const {
        if (cooldown_sec_ <= 0) return false;
#if defined(lag_unordered_ns)
        typename lag_unordered_ns::unordered_map<std::string, std::time_t>::const_iterator it = cooldown_map_.find(uid);
#else
        typename std::map<std::string, std::time_t>::const_iterator it = cooldown_map_.find(uid);
#endif
        if (it == cooldown_map_.end()) return false;
        return (nowSec - it->second) < cooldown_sec_;
    }

    static bool IsInList(const std::string& token, const std::vector<std::string>& lst) {
        for (size_t i = 0; i < lst.size(); ++i)
            if (token.find(lst[i]) != std::string::npos) return true;
        return false;
    }

    static bool ContainsAnyToken(const std::vector<std::string>& tokens,
                                 const std::vector<std::string>& lst) {
        for (size_t t=0;t<tokens.size();++t)
            if (IsInList(tokens[t], lst)) return true;
        return false;
    }

    void MaybeCleanupCooldown(std::time_t now) {
        if (cleanup_interval_sec_ <= 0) return;
        if (last_cleanup_sec_ != 0 && (now - last_cleanup_sec_) < cleanup_interval_sec_) return;
        last_cleanup_sec_ = now;

        // 오래된 항목 삭제(쿨다운의 4배 이상 지난 것)
        const std::time_t ttl = cooldown_sec_ > 0 ? cooldown_sec_ * 4 : 240;
#if defined(lag_unordered_ns)
        for (typename lag_unordered_ns::unordered_map<std::string, std::time_t>::iterator it = cooldown_map_.begin();
             it != cooldown_map_.end(); ) {
            if ((now - it->second) > ttl) {
                typename lag_unordered_ns::unordered_map<std::string, std::time_t>::iterator eras = it++;
                cooldown_map_.erase(eras);
            } else {
                ++it;
            }
        }
#else
        for (std::map<std::string, std::time_t>::iterator it = cooldown_map_.begin();
             it != cooldown_map_.end(); ) {
            if ((now - it->second) > ttl) {
                std::map<std::string, std::time_t>::iterator eras = it++;
                cooldown_map_.erase(eras);
            } else {
                ++it;
            }
        }
#endif
    }
};

} // namespace LagComplaint

#endif // LAG_COMPLAINT_DETECTOR_H_
