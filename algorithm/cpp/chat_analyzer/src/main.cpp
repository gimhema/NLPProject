#include <iostream>
#include <vector>
#include <ctime>
#include "ChatDetector.h"

// (1) 로컬 타입 금지 → 전역(네임스페이스) 범위로 이동
struct Msg {
    std::string uid;
    std::string text;
    int dt;
    // (2) C++03 호환을 위한 간단한 생성자 제공
    Msg(const std::string& u, const std::string& t, int d)
        : uid(u), text(t), dt(d) {}
};

int main() {
    using namespace LagComplaint;

    Detector det;
    det.SetCooldownSeconds(60);   // 같은 유저 60초 쿨타임
    det.SetEwmaLambda(0.95);      // 완만한 EWMA
    det.SetCusum(0.2, 3.0);       // CUSUM 민감도/임계

    std::vector<Msg> msgs;
    // (3) C99 compound literal 대신 생성자 사용
    msgs.push_back(Msg("u1", "렉이 걸리네 자꾸 끊김", 0));
    msgs.push_back(Msg("u2", "느리다 진짜 반응이 너무 느려요", 1));
    msgs.push_back(Msg("u1", "아님 내 인터넷 문제였음", 2)); // negation
    msgs.push_back(Msg("u3", "ping spike? lag?", 3));          // 질문 패널티
    msgs.push_back(Msg("u2", "프레임 드랍 심함", 4));
    msgs.push_back(Msg("u4", "내 와이파이 문제 같음 괜찮음", 5));
    msgs.push_back(Msg("u5", "서버 터짐? 계속 멈춤", 6));

    std::time_t t0 = std::time(0);

    double bucket = 0.0;
    for (size_t i = 0; i < msgs.size(); ++i) {
        std::time_t now = t0 + msgs[i].dt;
        DetectResult r = det.Classify(msgs[i].text, msgs[i].uid, now);
        if (r.is_complaint) bucket += r.weight;

        det.UpdateMetrics(bucket, now);

        std::cout << "[" << msgs[i].uid << "] \"" << msgs[i].text << "\" -> "
                  << "complaint=" << (r.is_complaint ? "Y" : "N")
                  << ", hits=" << r.hits
                  << ", weight=" << r.weight
                  << ", EWMA=" << det.EwmaScore()
                  << ", CUSUM_ALARM=" << (det.CusumAlarm() ? "Y" : "N")
                  << std::endl;

        bucket = 0.0;
    }

    return 0;
}
