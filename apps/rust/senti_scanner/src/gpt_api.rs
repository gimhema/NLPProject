// src/gpt_api.rs

use std::{env, error::Error};

use reqwest::blocking::Client;
use serde_json::{json, Value};

/// OpenAI Responses API 를 호출해서 모델 응답을 String 으로 반환.
/// 실패 시 에러를 반환.
pub fn call_gpt(prompt: &str) -> Result<String, Box<dyn Error>> {
    // 1) 환경 변수에서 API 키 읽기
    let api_key =
        env::var("OPENAI_API_KEY").expect("Please set the OPENAI_API_KEY environment variable.");

    // 2) HTTP 클라이언트 생성
    let client = Client::new();

    // 3) 요청 바디 구성
    let body = json!({
        "model": "gpt-5.1-mini",
        "input": prompt,
        "max_output_tokens": 100
    });

    // 4) HTTP POST 요청 보내기
    let res = client
        .post("https://api.openai.com/v1/responses")
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()?; // 네트워크 에러시 Err

    // 5) HTTP 상태 코드 체크
    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().unwrap_or_else(|_| "<body 읽기 실패>".to_string());
        return Err(format!("HTTP 오류: {status}, 응답 바디: {text}").into());
    }

    // 6) JSON 응답 파싱
    let v: Value = res.json()?;

    // Responses API 기본 구조에 맞춰 텍스트 추출
    if let Some(text) = v["output"][0]["content"][0]["text"].as_str() {
        Ok(text.to_owned())
    } else if let Some(text) = v["output_text"].as_str() {
        Ok(text.to_owned())
    } else {
        // 구조가 바뀌었거나 예상과 다를 때 전체 JSON을 문자열로 돌려줌
        Ok(format!("예상치 못한 응답 구조입니다. 전체 JSON: {v}"))
    }
}
