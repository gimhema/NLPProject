from typing import Any, Dict
import json

from sklearn.svm import LinearSVC


def train_linear_svm(X, y, C: float = 1.0) -> LinearSVC:
    """
    X : TF-IDF feature matrix
    y : labels
    """
    clf = LinearSVC(C=C)
    clf.fit(X, y)
    return clf


def svm_to_export_dict(model: LinearSVC) -> Dict[str, Any]:
    """
    LinearSVC도 선형 모델이므로 weight, bias를 그대로 export 가능.
    """
    classes = model.classes_.tolist()
    coef = model.coef_.tolist()
    intercept = model.intercept_.tolist()

    return {
        "type": "linear_svm",
        "classes": classes,
        "coef": coef,
        "intercept": intercept,
    }


def save_svm_model(model: LinearSVC, path: str):
    data = svm_to_export_dict(model)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
