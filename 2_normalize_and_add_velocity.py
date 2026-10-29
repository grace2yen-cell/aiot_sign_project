"""
2_normalize_and_add_velocity.py
[실제 사용] 학습 파이프라인 Step 2 - 속도 피처 추가
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 2: 상대 좌표 정규화 + Velocity 피처 추가
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목표:
- 이미 Step 1에서 상대 좌표 변환 완료 (어깨 기준)
- Velocity 피처 추가 (프레임 간 차분)
- 최종 차원: 168 → 336 (원본 168 + 속도 168)

입력:
- processed_data/클래스명/train_raw.npy: (N, 30, 168)
- processed_data/클래스명/val_raw.npy: (M, 30, 168)

출력:
- processed_data/클래스명/train_normalized.npy: (N, 30, 336)
- processed_data/클래스명/val_normalized.npy: (M, 30, 336)

핵심:
- Velocity = frame[t] - frame[t-1]
- 첫 프레임의 velocity는 0
- up/down 방향 구분에 매우 중요!
"""

import numpy as np
import os
from config import *


# ==================== Velocity 피처 추가 ====================
def add_velocity_features(sequences):
    """
    프레임 간 속도(velocity) 피처 추가

    Parameters:
    -----------
    sequences: np.array
        원본 시퀀스 (N, 30, 168)

    Returns:
    --------
    enhanced_sequences: np.array
        원본 + 속도 (N, 30, 336)
    """
    print("\n[Velocity] 속도 피처 계산 중...")

    # 프레임 간 차분 계산
    # velocities[t] = sequences[t+1] - sequences[t]
    velocities = np.diff(sequences, axis=1)  # (N, 29, 168)

    # 첫 프레임의 속도는 0으로 패딩
    first_frame_vel = np.zeros((sequences.shape[0], 1, sequences.shape[2]))  # (N, 1, 168)
    velocities = np.concatenate([first_frame_vel, velocities], axis=1)  # (N, 30, 168)

    # 원본 + 속도 결합
    enhanced_sequences = np.concatenate([sequences, velocities], axis=2)  # (N, 30, 336)

    print("Velocity 추가 완료!")
    print(f"   원본 shape: {sequences.shape}")
    print(f"   속도 shape: {velocities.shape}")
    print(f"   결합 shape: {enhanced_sequences.shape}")

    return enhanced_sequences


# ==================== 속도 패턴 분석 (디버깅용) ====================
def analyze_velocity_patterns(sequences, labels, class_idx_1, class_idx_2):
    """
    두 클래스의 속도 패턴 비교

    예: tempup vs tempdown의 Y 방향 속도 비교

    Parameters:
    -----------
    sequences: np.array (N, 30, 168)
    labels: np.array (N,)
    class_idx_1: int
    class_idx_2: int
    """
    if len(sequences) == 0:
        return

    print(f"\n{CLASS_NAMES[class_idx_1]} vs {CLASS_NAMES[class_idx_2]} 속도 패턴 분석...")

    # 클래스별 데이터 추출
    class1_seqs = sequences[labels == class_idx_1]
    class2_seqs = sequences[labels == class_idx_2]

    if len(class1_seqs) == 0 or len(class2_seqs) == 0:
        print("데이터 부족")
        return

    # 속도 계산 (프레임 간 차이)
    class1_vel = np.diff(class1_seqs, axis=1)  # (N, 29, 168)
    class2_vel = np.diff(class2_seqs, axis=1)

    # Y 좌표 속도만 추출
    # Hands: index 1, 4, 7, ... (양손 Y 좌표)
    # 양손 첫 21*2=42개 포인트의 Y 값
    y_indices = [i for i in range(1, 126, 3)]  # Y 좌표 인덱스

    class1_vel_y = class1_vel[:, :, y_indices]
    class2_vel_y = class2_vel[:, :, y_indices]

    # 평균 Y 속도
    class1_avg_vel_y = class1_vel_y.mean()
    class2_avg_vel_y = class2_vel_y.mean()

    print(f"\n   {CLASS_NAMES[class_idx_1]:12s}: {class1_avg_vel_y:+.6f}")
    print(f"   {CLASS_NAMES[class_idx_2]:12s}: {class2_avg_vel_y:+.6f}")

    # 방향 확인
    if class1_avg_vel_y * class2_avg_vel_y < 0:
        print("   속도 방향이 반대! (구분 가능)")
    else:
        print("   속도 방향이 같음 (확인 필요)")


# ==================== 메인 처리 ====================
def process_all_classes():
    """모든 클래스에 대해 정규화 + Velocity 추가"""

    print("=" * 60)
    print("Step 2: 정규화 + Velocity 피처 추가")
    print("=" * 60)

    for class_name in CLASS_NAMES:
        print(f"\n{'=' * 60}")
        print(f"클래스: {class_name}")
        print(f"{'=' * 60}")

        class_dir = os.path.join(PROCESSED_DIR, class_name)

        # ===== Train 처리 =====
        train_raw_path = os.path.join(class_dir, "train_raw.npy")

        if os.path.exists(train_raw_path):
            print("\n[Train] 처리 중...")

            # 로드
            train_raw = np.load(train_raw_path)  # (N, 30, 168)
            print(f"   원본: {train_raw.shape}")

            # Velocity 추가
            train_normalized = add_velocity_features(train_raw)  # (N, 30, 336)

            # 저장
            train_norm_path = os.path.join(class_dir, "train_normalized.npy")
            np.save(train_norm_path, train_normalized)

            print(f"저장: {train_normalized.shape}")
            print(f"   {train_norm_path}")
        else:
            print(f"\nTrain 데이터 없음: {train_raw_path}")

        # ===== Val 처리 =====
        val_raw_path = os.path.join(class_dir, "val_raw.npy")

        if os.path.exists(val_raw_path):
            print("\n[Val] 처리 중...")

            # 로드
            val_raw = np.load(val_raw_path)  # (M, 30, 168)
            print(f"   원본: {val_raw.shape}")

            # Velocity 추가
            val_normalized = add_velocity_features(val_raw)  # (M, 30, 336)

            # 저장
            val_norm_path = os.path.join(class_dir, "val_normalized.npy")
            np.save(val_norm_path, val_normalized)

            print(f"저장: {val_normalized.shape}")
            print(f"   {val_norm_path}")
        else:
            print(f"\nVal 데이터 없음: {val_raw_path}")

    print("\n" + "=" * 60)
    print("정규화 + Velocity 완료!")
    print("=" * 60)
    print(f"저장 위치: {PROCESSED_DIR}")
    print("최종 차원: (30, 336) = (30, 168 원본 + 168 속도)")
    print("=" * 60)


# ==================== Velocity 검증 (선택사항) ====================
def verify_velocity():
    """Velocity 방향이 올바른지 검증"""

    print("\n" + "=" * 60)
    print("Velocity 방향 검증")
    print("=" * 60)

    # Train 데이터로만 검증
    all_train_seqs = []
    all_train_labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(PROCESSED_DIR, class_name)
        train_raw_path = os.path.join(class_dir, "train_raw.npy")

        if os.path.exists(train_raw_path):
            train_raw = np.load(train_raw_path)
            all_train_seqs.append(train_raw)
            all_train_labels.extend([class_idx] * len(train_raw))

    if len(all_train_seqs) == 0:
        print("검증 데이터 없음")
        return

    all_train_seqs = np.concatenate(all_train_seqs, axis=0)
    all_train_labels = np.array(all_train_labels)

    # tempup vs tempdown 분석
    if "tempup" in CLASS_NAMES and "tempdown" in CLASS_NAMES:
        tempup_idx = CLASS_MAPPING["tempup"]
        tempdown_idx = CLASS_MAPPING["tempdown"]
        analyze_velocity_patterns(all_train_seqs, all_train_labels, tempup_idx, tempdown_idx)

    # soundup vs sounddown 분석
    if "soundup" in CLASS_NAMES and "sounddown" in CLASS_NAMES:
        soundup_idx = CLASS_MAPPING["soundup"]
        sounddown_idx = CLASS_MAPPING["sounddown"]
        analyze_velocity_patterns(all_train_seqs, all_train_labels, soundup_idx, sounddown_idx)

    # blindup vs blinddown 분석
    if "blindup" in CLASS_NAMES and "blinddown" in CLASS_NAMES:
        blindup_idx = CLASS_MAPPING["blindup"]
        blinddown_idx = CLASS_MAPPING["blinddown"]
        analyze_velocity_patterns(all_train_seqs, all_train_labels, blindup_idx, blinddown_idx)

    print("=" * 60)


# ==================== 실행 ====================
if __name__ == "__main__":
    # 정규화 + Velocity 추가
    process_all_classes()

    # Velocity 검증 (선택사항)
    verify_velocity()

    print("\n다음 단계: 3_check_data.py 실행")
