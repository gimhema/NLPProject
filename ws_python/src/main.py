import argparse
import csv
import os
from typing import List, Tuple

from tf_idf import TfidfTrainer, TfidfConfig
from logistic_reg import (
    train_logistic_regression,
    save_logistic_model,
    load_logistic_model,
)
from svm import (
    train_linear_svm,
    save_svm_model,
    load_svm_model,
)

# from plot_utils import plot_class_distribution
from sklearn.decomposition import PCA
import numpy as np
from plot_utils import plot_svm_decision_boundary


def load_dataset_csv(path: str) -> Tuple[List[str], List[str]]:
    """
    CSV 포맷: category,text (첫 줄은 헤더)
    """
    labels: List[str] = []
    texts: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
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


def test_classifier(tfidf_path: str, model_path: str, algo: str, test_texts: List[str]):
    """
    학습된 TF-IDF / 모델을 로드해서 문장을 테스트한다.
    """
    print("\n[TEST] Running classifier tests...")

    # TF-IDF 로드
    tfidf_trainer = TfidfTrainer(TfidfConfig())
    tfidf_trainer.load_json(tfidf_path)

    # 모델 로드
    if algo == "logreg":
        model = load_logistic_model(model_path)
    else:
        model = load_svm_model(model_path)

    # 각 문장 테스트
    for text in test_texts:
        vec = tfidf_trainer.transform([text])
        pred = model.predict(vec)[0]
        print(f"  [INPUT] {text}")
        print(f"  --> Predicted Category: {pred}\n")


# ---------------------------------------------------------
#  main
# ---------------------------------------------------------
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1) 데이터 로드
    if args.data:
        labels, texts = load_dataset_csv(args.data)
        if not labels:
            print("[WARN] Dataset is empty, fallback to toy dataset.")
            # labels, texts = build_toy_dataset()
    else:
        print("[INFO] No dataset path given. Using toy dataset.")
        # labels, texts = build_toy_dataset()

    # 2) TF-IDF 학습
    config = TfidfConfig(ngram_min=1, ngram_max=args.ngram_max)
    tfidf_trainer = TfidfTrainer(config)
    X = tfidf_trainer.fit_transform(texts)

    # 출력 폴더 생성
    os.makedirs(args.out_dir, exist_ok=True)
    tfidf_path = os.path.join(args.out_dir, "tfidf.json")

    # TF-IDF 저장
    print(f"[INFO] Training TF-IDF (ngram_max={args.ngram_max}) ...")
    tfidf_trainer.save_json(tfidf_path)
    print(f"[INFO] Saved TF-IDF model to {tfidf_path}")

    # 3) 알고리즘 선택 & 학습
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

    print("[INFO] Training Done.")

    test_samples = [
        ""
    ]

    test_classifier(tfidf_path, model_path, args.algo, test_samples)

    X_2d = PCA(n_components=2).fit_transform(X.toarray())
    plot_svm_decision_boundary(X_2d, labels, model)


if __name__ == "__main__":
    main()
