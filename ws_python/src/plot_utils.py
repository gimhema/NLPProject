# plot_utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_svm_decision_boundary(X, y, model, feature_names=None, title="SVM Decision Boundary"):
    """
    LinearSVC 모델의 2차원 feature에 대한 결정 경계를 시각화
    - X: (n_samples, 2) numpy array
    - y: labels (array-like)
    - model: 학습된 LinearSVC
    """
    if X.shape[1] != 2:
        raise ValueError("X must have exactly 2 features for plotting.")

    # 결정 경계 그리기
    coef = model.coef_[0]
    intercept = model.intercept_[0]

    # x1 범위
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    Z = coef[0]*xx + coef[1]*yy + intercept
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.2, colors=["blue", "red"])
    plt.scatter(X[:, 0], X[:, 1], c=[0 if label==model.classes_[0] else 1 for label in y], cmap=plt.cm.bwr, edgecolors='k')

    plt.xlabel(feature_names[0] if feature_names else "Feature 1")
    plt.ylabel(feature_names[1] if feature_names else "Feature 2")
    plt.title(title)
    plt.show()
