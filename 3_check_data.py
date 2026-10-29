"""
3_check_data.py
[실제 사용] 학습 파이프라인 Step 3 - 증강 전 데이터 검증
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 3: 데이터 검증
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목표:
- 각 클래스별 Train/Val 데이터 개수 확인
- Shape 일관성 확인
- NaN/Inf 체크
- 데이터 품질 확인

체크 항목:
1. 개수 확인 (Train ~20개, Val ~5개)
2. Shape 확인 ((N, 30, 336))
3. NaN/Inf 없음 확인
4. 통계량 확인 (평균, 분산)
"""

import numpy as np
import os
from config import *


# ==================== 데이터 검증 ====================
def check_data_quality():
    """모든 클래스 데이터 품질 검사"""

    print("=" * 60)
    print("Step 3: 데이터 검증")
    print("=" * 60)

    print("\n예상 데이터:")
    print("   Train: person1~4 → 약 16~28개/클래스 (수화)")
    print("                    → 약 160개 (none)")
    print("   Val:   person5   → 약 4~7개/클래스 (수화)")
    print("                    → 약 40개 (none)")
    print("   Shape: (N, 30, 336)")
    print("   차원: 336 = 168 원본 + 168 속도")

    # 결과 저장
    results = {"train": {}, "val": {}}

    total_train = 0
    total_val = 0

    print("\n" + "=" * 60)
    print("클래스별 데이터 현황")
    print("=" * 60)
    print(f"{'Class':<12} {'Train':<15} {'Val':<15} {'Shape':<20}")
    print("-" * 60)

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(PROCESSED_DIR, class_name)

        # Train
        train_norm_path = os.path.join(class_dir, "train_normalized.npy")
        train_count = 0
        train_shape = "N/A"

        if os.path.exists(train_norm_path):
            train_data = np.load(train_norm_path)
            train_count = len(train_data)
            train_shape = str(train_data.shape)
            total_train += train_count

            results["train"][class_name] = {
                "count": train_count,
                "shape": train_data.shape,
                "has_nan": np.isnan(train_data).any(),
                "has_inf": np.isinf(train_data).any(),
            }

        # Val
        val_norm_path = os.path.join(class_dir, "val_normalized.npy")
        val_count = 0
        val_shape = "N/A"

        if os.path.exists(val_norm_path):
            val_data = np.load(val_norm_path)
            val_count = len(val_data)
            val_shape = str(val_data.shape)
            total_val += val_count

            results["val"][class_name] = {
                "count": val_count,
                "shape": val_data.shape,
                "has_nan": np.isnan(val_data).any(),
                "has_inf": np.isinf(val_data).any(),
            }

        # 출력
        print(
            f"{class_name:<12} {train_count:<15} {val_count:<15} "
            f"{train_shape if train_count > 0 else val_shape}"
        )

    print("-" * 60)
    print(f"{'Total':<12} {total_train:<15} {total_val:<15}")
    print("=" * 60)

    # ===== NaN/Inf 체크 =====
    print("\n" + "=" * 60)
    print("NaN/Inf 체크")
    print("=" * 60)

    has_issue = False

    for split in ["train", "val"]:
        for class_name in results[split]:
            info = results[split][class_name]

            if info["has_nan"]:
                print(f"{split.upper()} - {class_name}: NaN 발견!")
                has_issue = True

            if info["has_inf"]:
                print(f"{split.upper()} - {class_name}: Inf 발견!")
                has_issue = True

    if not has_issue:
        print("NaN/Inf 없음 - 정상!")

    # ===== Shape 일관성 체크 =====
    print("\n" + "=" * 60)
    print("Shape 일관성 체크")
    print("=" * 60)

    expected_shape = (SEQUENCE_LENGTH, FEATURE_DIM_FINAL)
    shape_issue = False

    for split in ["train", "val"]:
        for class_name in results[split]:
            info = results[split][class_name]
            actual_shape = info["shape"][1:]  # (30, 336)

            if actual_shape != expected_shape:
                print(
                    f"{split.upper()} - {class_name}: "
                    f"Shape 불일치! Expected: {expected_shape}, Got: {actual_shape}"
                )
                shape_issue = True

    if not shape_issue:
        print("모든 데이터 Shape 일관성 확인!")
        print(f"   Expected: {expected_shape}")

    # ===== 통계량 확인 =====
    print("\n" + "=" * 60)
    print("통계량 확인 (Train 기준)")
    print("=" * 60)

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(PROCESSED_DIR, class_name)
        train_norm_path = os.path.join(class_dir, "train_normalized.npy")

        if not os.path.exists(train_norm_path):
            continue

        data = np.load(train_norm_path)

        # 원본 부분 (0:168)
        original = data[:, :, :FEATURE_DIM_RAW]

        # 속도 부분 (168:336)
        velocity = data[:, :, FEATURE_DIM_RAW:]

        print(f"\n{class_name}:")
        print(f"   원본 - Mean: {original.mean():.6f}, Std: {original.std():.6f}")
        print(f"   속도 - Mean: {velocity.mean():.6f}, Std: {velocity.std():.6f}")
        print(f"   속도 범위: [{velocity.min():.6f}, {velocity.max():.6f}]")

    # ===== 권장 사항 =====
    print("\n" + "=" * 60)
    print("권장 사항")
    print("=" * 60)

    if total_train < 100:
        print("Train 데이터가 너무 적습니다 (< 100개)")
        print("   → 증강 필요!")
    else:
        print(f"Train 데이터: {total_train}개 (양호)")

    if total_val < 40:
        print("Val 데이터가 너무 적습니다 (< 40개)")
    else:
        print(f"Val 데이터: {total_val}개 (양호)")

    # None 클래스 체크
    if "none" in results["train"]:
        none_train = results["train"]["none"]["count"]
        if none_train < 100:
            print(f"None Train 데이터 부족 ({none_train}개 < 100개)")

    print("=" * 60)

    return results


# ==================== 증강 필요성 판단 ====================
def check_augmentation_need():
    """증강이 필요한지 판단"""

    print("\n" + "=" * 60)
    print("증강 필요성 판단")
    print("=" * 60)

    print("\n증강 목표:")
    print(f"   각 수화 클래스: {AUGMENT_TARGET['sign_train']}개")
    print(f"   None 클래스: {AUGMENT_TARGET['none_train']}개")
    print(f"   총 Train: {AUGMENT_TARGET['sign_train'] * 8 + AUGMENT_TARGET['none_train']}개")

    print("\n현재 상태:")

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(PROCESSED_DIR, class_name)
        train_norm_path = os.path.join(class_dir, "train_normalized.npy")

        if os.path.exists(train_norm_path):
            data = np.load(train_norm_path)
            current_count = len(data)

            if class_name == "none":
                target = AUGMENT_TARGET["none_train"]
            else:
                target = AUGMENT_TARGET["sign_train"]

            multiplier = target / current_count if current_count > 0 else 0

            print(
                f"   {class_name:<12}: {current_count:4d}개 → {target:4d}개 " f"(x{multiplier:.1f})"
            )

    print("\n다음 단계: 4_augment_data.py 실행하여 데이터 증강")
    print("=" * 60)


# ==================== 실행 ====================
if __name__ == "__main__":
    # 데이터 검증
    results = check_data_quality()

    # 증강 필요성 판단
    check_augmentation_need()

    print("\n검증 완료!")
