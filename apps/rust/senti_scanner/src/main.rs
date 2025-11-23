mod data_loader;
mod gpt_api;
mod offline_algo;
mod offline_emotion;
mod scanner_mode;

use offline_emotion::OfflineCategoryEngine;
use offline_algo::Category;

fn main() {
    // 에러를 깔끔하게 처리하기 위해 run()으로 분리
    if let Err(e) = run() {
        eprintln!("오류 발생: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // 여기서 원하는 프롬프트를 구성해서 gpt_api 모듈에 넘김
    let prompt = "Hello, GPT! Can you provide a brief introduction about yourself in Korean?";

    let reply = gpt_api::call_gpt(prompt)?;
    println!("모델 응답:\n{}", reply);

    Ok(())
}


/*


fn main() {
    let mut engine = OfflineCategoryEngine::new();

    // 학습 데이터가 있다면 여기서 로드
    // 예: "data/training.tsv"
    let _ = engine.train_from_tsv("data/training.tsv", 5, 20);

    let msg = "강화 실패해서 골드가 너무 없어서 힘들어";

    let result = engine.analyze(msg);

    println!("input: {}", msg);
    for (cat, score) in result {
        println!("{:?}: score={:.3}", cat, score);
    }
}

*/

/*

fn main() {
    // 카테고리 엔진 초기화 + 학습
    let mut cat_engine = OfflineCategoryEngine::new();
    let _ = cat_engine.train_from_tsv("data/category_training.tsv", 5, 20);

    // 감정 엔진 초기화 (규칙 기반이라 학습 불필요)
    let sent_engine = OfflineSentimentEngine::new();

    let msg = "강화 실패해서 골드가 너무 없어서 힘들어";

    let cat_result = cat_engine.analyze(msg);
    let sent_result = sent_engine.analyze(msg);
    let top_sent = sent_engine.top_sentiment(msg);

    println!("input: {}", msg);

    println!("== Category scores ==");
    for (cat, score) in cat_result {
        println!("{:?}: score={:.3}", cat, score);
    }

    println!("== Sentiment scores ==");
    for (s, score) in sent_result {
        println!("{:?}: score={}", s, score);
    }
    println!("Top sentiment: {:?}", top_sent);
}
*/