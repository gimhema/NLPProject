from typing import Any, Dict
import json

from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X, y, C: float = 1.0, max_iter: int = 1000) -> LogisticRegression:
    """
    X : TF-IDF feature matrix (scipy sparse)
    y : list[str] or array-like of labels
    """
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class="auto",
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def logistic_to_export_dict(model: LogisticRegression) -> Dict[str, Any]:
    """
    C++에서 그대로 weight, bias를 사용할 수 있도록 export용 dict 생성.
    - classes: 카테고리 이름 리스트
    - coef: shape (num_classes, num_features)
    - intercept: shape (num_classes,)
    """
    classes = model.classes_.tolist()
    coef = model.coef_.tolist()
    intercept = model.intercept_.tolist()

    return {
        "type": "logistic_regression",
        "classes": classes,
        "coef": coef,
        "intercept": intercept,
    }


def save_logistic_model(model: LogisticRegression, path: str):
    data = logistic_to_export_dict(model)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
