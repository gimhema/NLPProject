mod gpt_api;

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
