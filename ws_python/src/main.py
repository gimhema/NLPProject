import argparse
import csv
import os
from typing import List, Tuple

from tf_idf import TfidfTrainer, TfidfConfig
from logistic_reg import train_logistic_regression, save_logistic_model
from svm import train_linear_svm, save_svm_model


def load_dataset_csv(path: str) -> Tuple[List[str], List[str]]:
    """
    CSV 포맷: category,text
    (헤더 유무는 옵션으로 둘 수 있는데, 여기서는 첫 줄을 헤더로 가정)
    """
    labels: List[str] = []
    texts: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 헤더: category, text
        for row in reader:
            label = row.get("category")
            text = row.get("text")
            if not label or not text:
                continue
            labels.append(label)
            texts.append(text)

    return labels, texts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline chat classifier trainer")
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        help="Path to CSV dataset (columns: category,text).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["logreg", "svm"],
        default="logreg",
        help="Which algorithm to train.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="model_out",
        help="Output directory for exported JSON files.",
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=1,
        help="Max n-gram size for TF-IDF.",
    )
    return parser


def build_toy_dataset() -> Tuple[List[str], List[str]]:
    """
    사용자가 예시로 준 작은 데이터셋 (CSV 없을 때 fallback)
    """
    pairs = [
        ("Capital", "나는 빌게이츠처럼 많은 돈을 벌고싶어"),
        ("Capital", "와 엔비디아 주식 오른거봐라 젠슨황 돈 많이 벌었겠네"),
        ("Battle", "최종 레이드 던전에 도전할수있을만큼 강해지고싶어"),
        ("Battle", "새로나온 인던에 도전하기엔 내 스펙이 많이 모잘라"),
        ("Pet", "이번에 출시된 드래곤 펫의 디자인이 꽤 귀엽군"),
        ("Capital", "베조스 아마존 주식 대박 ㄷㄷ"),
    ]
    labels = [c for (c, _) in pairs]
    texts = [t for (_, t) in pairs]
    return labels, texts


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1) 데이터 로드
    if args.data:
        labels, texts = load_dataset_csv(args.data)
        if not labels:
            print("[WARN] Dataset is empty, fallback to toy dataset.")
            labels, texts = build_toy_dataset()
    else:
        print("[INFO] No dataset path given. Using toy dataset.")
        labels, texts = build_toy_dataset()

    # 2) TF-IDF 학습
    config = TfidfConfig(
        ngram_min=1,
        ngram_max=args.ngram_max,
    )
    tfidf_trainer = TfidfTrainer(config)
    X = tfidf_trainer.fit_transform(texts)

    # 3) 알고리즘 선택 & 학습
    os.makedirs(args.out_dir, exist_ok=True)
    tfidf_path = os.path.join(args.out_dir, "tfidf.json")

    print(f"[INFO] Training TF-IDF (ngram_max={args.ngram_max}) ...")
    tfidf_trainer.save_json(tfidf_path)
    print(f"[INFO] Saved TF-IDF model to {tfidf_path}")

    if args.algo == "logreg":
        print("[INFO] Training Logistic Regression ...")
        model = train_logistic_regression(X, labels)
        model_path = os.path.join(args.out_dir, "logistic_regression.json")
        save_logistic_model(model, model_path)
        print(f"[INFO] Saved Logistic Regression model to {model_path}")
    else:
        print("[INFO] Training Linear SVM ...")
        model = train_linear_svm(X, labels)
        model_path = os.path.join(args.out_dir, "linear_svm.json")
        save_svm_model(model, model_path)
        print(f"[INFO] Saved Linear SVM model to {model_path}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
