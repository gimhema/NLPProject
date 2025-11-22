use std::fs::File;
use std::io::{self, BufRead, BufReader};

use crate::offline_algo::{Category, LabeledSample};

/// 문자열을 Category enum으로 변환.
/// 파일에 기록할 때 "capital", "growth" 이런 식으로 써두면 됨.
fn parse_category(s: &str) -> Option<Category> {
    match s.trim() {
        "capital" => Some(Category::Capital),
        "growth"  => Some(Category::Growth),
        "craft"   => Some(Category::Craft),
        "party"   => Some(Category::Party),
        "misc"    => Some(Category::Misc),
        _ => None,
    }
}

/// TSV 파일에서 학습 데이터 로드
/// 한 줄 형식: <category>\t<text>
pub fn load_labeled_samples_from_tsv(path: &str) -> io::Result<Vec<LabeledSample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut samples: Vec<LabeledSample> = Vec::new();

    for line_res in reader.lines() {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }

        let mut split = line.splitn(2, '\t');
        let cat_str = match split.next() {
            Some(s) => s,
            None => continue,
        };
        let text_str = match split.next() {
            Some(s) => s,
            None => continue,
        };

        if let Some(cat) = parse_category(cat_str) {
            samples.push(LabeledSample {
                category: cat,
                text: text_str.to_string(),
            });
        }
    }

    Ok(samples)
}
