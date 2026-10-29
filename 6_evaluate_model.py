"""
6_evaluate_model.py
[실제 사용] 학습 파이프라인 Step 6 - Val 데이터로 모델 평가
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 6: 모델 평가 (Val 데이터 기준)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목표:
- 학습된 모델의 성능을 Val 데이터로 평가
- Confusion Matrix 생성
- 클래스별 성능 지표 계산
- 전체 평균 정확도 및 신뢰도 계산

입력:
- models/lstm_model.h5: 학습된 모델
- augmented_data/X_val.npy: Val 데이터
- augmented_data/y_val.npy: Val 라벨

출력:
- visualization/evaluation/confusion_matrix.png
- visualization/evaluation/confusion_matrix_normalized.png
- visualization/evaluation/confidence_distribution.png
- visualization/evaluation/evaluation_report.txt

지표:
- 클래스별 정확도 (Accuracy)
- 클래스별 Precision, Recall, F1-Score
- 전체 평균 정확도
- 전체 평균 신뢰도
"""

import gpu_config  # GPU 설정

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from config import *


# ==================== 데이터 로드 ====================
def load_validation_data():
    """Val 데이터 로드 및 정규화"""

    print("\n[1/5] 데이터 로드 중...")

    X_val = np.load(os.path.join(AUGMENTED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(AUGMENTED_DIR, "y_val.npy"))

    print(f"Val 데이터: {X_val.shape}")
    print(f"   샘플 수: {len(X_val)}")

    # 정규화 파라미터 로드
    norm_params_path = os.path.join(AUGMENTED_DIR, "normalization_params.npz")

    if os.path.exists(norm_params_path):
        params = np.load(norm_params_path)
        mean = params["mean"]
        std = params["std"]

        X_val_norm = (X_val - mean) / std
        print("정규화 완료")
    else:
        X_val_norm = X_val
        print("정규화 파라미터 없음")

    # One-hot encoding
    Y_val = to_categorical(y_val, NUM_CLASSES)

    return X_val_norm, y_val, Y_val


# ==================== 모델 평가 ====================
def evaluate_model(model, X_val, Y_val):
    """모델 평가"""

    print("\n[2/5] 모델 평가 중...")

    loss, accuracy = model.evaluate(X_val, Y_val, verbose=0)

    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    return loss, accuracy


# ==================== 예측 수행 ====================
def get_predictions(model, X_val):
    """예측 및 신뢰도 계산"""

    print("\n[3/5] 예측 수행 중...")

    # 예측 확률
    y_pred_proba = model.predict(X_val, verbose=0)

    # 예측 클래스
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 예측 신뢰도 (최대 확률값)
    y_pred_confidence = np.max(y_pred_proba, axis=1)

    print(f"예측 완료: {len(y_pred)} 샘플")

    return y_pred, y_pred_proba, y_pred_confidence


# ==================== Confusion Matrix 시각화 ====================
def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Confusion Matrix 그리기"""

    print("\n[4/5] Confusion Matrix 생성 중...")

    os.makedirs(save_dir, exist_ok=True)

    # Confusion Matrix 계산
    cm = confusion_matrix(y_true, y_pred)

    # 정규화된 Confusion Matrix (행 기준)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # ===== 1) 절대값 Confusion Matrix =====
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_title("Confusion Matrix (Counts)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ===== 2) 정규화된 Confusion Matrix =====
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar_kws={"label": "Percentage"},
        vmin=0,
        vmax=1,
    )

    ax.set_title("Confusion Matrix (Normalized)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"), dpi=150)
    plt.close()

    print("Confusion Matrix 저장 완료")

    return cm, cm_normalized


# ==================== 신뢰도 분포 시각화 ====================
def plot_confidence_distribution(y_true, y_pred, y_pred_confidence, save_dir):
    """신뢰도 분포 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 전체 신뢰도 분포
    axes[0, 0].hist(y_pred_confidence, bins=50, color="skyblue", edgecolor="black")
    axes[0, 0].axvline(
        y_pred_confidence.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {y_pred_confidence.mean():.4f}",
    )
    axes[0, 0].set_title("Overall Confidence Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Confidence")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 정답/오답별 신뢰도
    correct = y_true == y_pred
    axes[0, 1].hist(
        [y_pred_confidence[correct], y_pred_confidence[~correct]],
        bins=30,
        label=["Correct", "Incorrect"],
        color=["green", "red"],
        alpha=0.6,
    )
    axes[0, 1].set_title("Confidence: Correct vs Incorrect", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Confidence")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 클래스별 평균 신뢰도
    class_confidences = [
        y_pred_confidence[y_true == i].mean() if len(y_pred_confidence[y_true == i]) > 0 else 0
        for i in range(NUM_CLASSES)
    ]

    axes[1, 0].bar(range(NUM_CLASSES), class_confidences, color="coral")
    axes[1, 0].set_title("Average Confidence by Class", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Class")
    axes[1, 0].set_ylabel("Average Confidence")
    axes[1, 0].set_xticks(range(NUM_CLASSES))
    axes[1, 0].set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # 4. 클래스별 정확도
    class_accuracies = [
        (y_pred[y_true == i] == i).mean() if len(y_pred[y_true == i]) > 0 else 0
        for i in range(NUM_CLASSES)
    ]

    axes[1, 1].bar(range(NUM_CLASSES), class_accuracies, color="lightgreen")
    axes[1, 1].set_title("Accuracy by Class", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Class")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_xticks(range(NUM_CLASSES))
    axes[1, 1].set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confidence_distribution.png"), dpi=150)
    plt.close()

    print("신뢰도 분포 저장 완료")


# ==================== 평가 리포트 생성 ====================
def save_evaluation_report(y_true, y_pred, y_pred_confidence, save_dir):
    """평가 리포트 txt 저장"""

    print("\n[5/5] 평가 리포트 생성 중...")

    os.makedirs(save_dir, exist_ok=True)

    report_path = os.path.join(save_dir, "evaluation_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("모델 평가 리포트 (Val 데이터 기준)\n")
        f.write("=" * 80 + "\n\n")

        # ===== 1) 전체 정확도 및 신뢰도 =====
        overall_accuracy = (y_true == y_pred).mean()
        overall_confidence = y_pred_confidence.mean()

        f.write("전체 성능\n")
        f.write("=" * 80 + "\n")
        f.write(f"전체 정확도 (Overall Accuracy): {overall_accuracy * 100:.2f}%\n")
        f.write(f"전체 평균 신뢰도 (Average Confidence): {overall_confidence:.4f}\n")
        f.write(f"총 샘플 수: {len(y_true)}\n")
        f.write(f"정답 수: {(y_true == y_pred).sum()}\n")
        f.write(f"오답 수: {(y_true != y_pred).sum()}\n")
        f.write("\n")

        # ===== 2) 클래스별 성능 =====
        f.write("클래스별 성능 분석\n")
        f.write("=" * 80 + "\n\n")

        # sklearn classification_report
        report = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=True
        )

        # 헤더
        f.write(
            f"{'Class':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
            f"{'F1-Score':<10} {'Support':<10} {'Avg Conf':<10}\n"
        )
        f.write("-" * 80 + "\n")

        # 클래스별 통계
        for i, class_name in enumerate(CLASS_NAMES):
            # 해당 클래스의 인덱스
            class_indices = np.where(y_true == i)[0]

            if len(class_indices) > 0:
                # 클래스별 정확도
                class_accuracy = (y_pred[class_indices] == i).mean()

                # 클래스별 평균 신뢰도
                avg_confidence = y_pred_confidence[class_indices].mean()

                # Precision, Recall, F1
                precision = report[class_name]["precision"]
                recall = report[class_name]["recall"]
                f1 = report[class_name]["f1-score"]
                support = report[class_name]["support"]

                f.write(
                    f"{class_name:<12} {class_accuracy:<10.4f} {precision:<10.4f} "
                    f"{recall:<10.4f} {f1:<10.4f} {support:<10.0f} {avg_confidence:<10.4f}\n"
                )

        f.write("-" * 80 + "\n")

        # 평균
        f.write(
            f"{'Macro Avg':<12} {overall_accuracy:<10.4f} "
            f"{report['macro avg']['precision']:<10.4f} "
            f"{report['macro avg']['recall']:<10.4f} "
            f"{report['macro avg']['f1-score']:<10.4f} "
            f"{len(y_true):<10.0f} {overall_confidence:<10.4f}\n"
        )

        f.write(
            f"{'Weighted Avg':<12} {overall_accuracy:<10.4f} "
            f"{report['weighted avg']['precision']:<10.4f} "
            f"{report['weighted avg']['recall']:<10.4f} "
            f"{report['weighted avg']['f1-score']:<10.4f} "
            f"{len(y_true):<10.0f} {overall_confidence:<10.4f}\n"
        )

        f.write("\n")

        # ===== 3) sklearn classification report (원본) =====
        f.write("=" * 80 + "\n")
        f.write("sklearn Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
        f.write("\n")

        # ===== 4) 오류 분석 =====
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]

        f.write("=" * 80 + "\n")
        f.write("오류 분석\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"총 오류 개수: {len(error_indices)} / {len(y_true)} "
            f"({len(error_indices) / len(y_true) * 100:.2f}%)\n\n"
        )

        if len(error_indices) > 0:
            # 가장 빈번한 오류 패턴
            f.write("가장 빈번한 오류 패턴:\n")
            f.write("-" * 80 + "\n")

            error_patterns = {}
            for idx in error_indices:
                true_class = CLASS_NAMES[y_true[idx]]
                pred_class = CLASS_NAMES[y_pred[idx]]
                pattern = f"{true_class} → {pred_class}"

                if pattern not in error_patterns:
                    error_patterns[pattern] = 0
                error_patterns[pattern] += 1

            sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)

            for i, (pattern, count) in enumerate(sorted_patterns[:10], 1):
                f.write(f"   {i}. {pattern:<30}: {count:3d}회\n")
        else:
            f.write("완벽한 예측! 오류 없음.\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"평가 리포트 저장: {report_path}")

    # 콘솔 출력
    print("\n" + "=" * 80)
    print("평가 결과 요약")
    print("=" * 80)
    print(f"전체 정확도: {overall_accuracy * 100:.2f}%")
    print(f"전체 평균 신뢰도: {overall_confidence:.4f}")
    print(f"오류 개수: {len(error_indices)} / {len(y_true)}")
    print("=" * 80)


# ==================== 메인 ====================
def main():
    print("=" * 80)
    print("Step 6: 모델 평가")
    print("=" * 80)

    eval_dir = os.path.join(VIZ_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # 1. 데이터 로드
    X_val, y_val, Y_val = load_validation_data()

    # 2. 모델 로드
    print(f"\n모델 로드 중: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일이 없습니다: {MODEL_PATH}")
        print("먼저 5_train_lstm.py를 실행하세요!")
        return

    model = load_model(MODEL_PATH)
    print("모델 로드 완료")

    # 3. 평가
    loss, accuracy = evaluate_model(model, X_val, Y_val)

    # 4. 예측
    y_pred, y_pred_proba, y_pred_confidence = get_predictions(model, X_val)

    # 5. Confusion Matrix
    cm, cm_normalized = plot_confusion_matrix(y_val, y_pred, eval_dir)

    # 6. 신뢰도 분포
    plot_confidence_distribution(y_val, y_pred, y_pred_confidence, eval_dir)

    # 7. 평가 리포트
    save_evaluation_report(y_val, y_pred, y_pred_confidence, eval_dir)

    print("\n" + "=" * 80)
    print("평가 완료!")
    print("=" * 80)
    print(f"결과 저장 위치: {eval_dir}")
    print("   - confusion_matrix.png")
    print("   - confusion_matrix_normalized.png")
    print("   - confidence_distribution.png")
    print("   - evaluation_report.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
