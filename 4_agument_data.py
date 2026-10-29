"""
4_augment_data.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 4: 데이터 증강 (Train + Val 모두)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 목표:
- Train 데이터 증강 (person1~4)
- Val 데이터 증강 (person5) ⭐ 추가!
- Train:Val = 80:20 비율 유지

📊 증강 목표:
- Train: 수화 600개/클래스, none 4800개
- Val: 수화 150개/클래스, none 1200개
- 총 Train 9,600개 : Val 2,400개

📊 출력:
- augmented_data/X_train.npy: (9600, 30, 336)
- augmented_data/y_train.npy: (9600,)
- augmented_data/X_val.npy: (2400, 30, 336)
- augmented_data/y_val.npy: (2400,)
"""

import numpy as np
import os
from tqdm import tqdm
from config import *


# ==================== 증강 함수들 ====================
def speed_change(sequence, speed_factor):
    """속도 변화"""
    T, D = sequence.shape
    new_length = int(T / speed_factor)

    if new_length < T // 2 or new_length > T * 2:
        new_length = T

    old_indices = np.linspace(0, T - 1, T)
    new_indices = np.linspace(0, T - 1, new_length)

    resampled = np.zeros((new_length, D))
    for d in range(D):
        resampled[:, d] = np.interp(new_indices, old_indices, sequence[:, d])

    if new_length != T:
        final_indices = np.linspace(0, new_length - 1, T, dtype=int)
        resampled = resampled[final_indices]

    return resampled


def add_temporal_noise(sequence, noise_level=TEMPORAL_NOISE_LEVEL):
    """시간축 노이즈 추가"""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise


def scale_sequence(sequence, scale_range=SPATIAL_SCALE_RANGE):
    """공간적 스케일 변화"""
    scale = np.random.uniform(scale_range[0], scale_range[1])

    T, D = sequence.shape
    scaled_seq = np.zeros_like(sequence)

    feature_dim = D // 2

    for t in range(T):
        frame_orig = sequence[t, :feature_dim].reshape(-1, 3)
        frame_vel = sequence[t, feature_dim:]

        center = np.mean(frame_orig, axis=0)
        scaled_frame = (frame_orig - center) * scale + center

        scaled_seq[t, :feature_dim] = scaled_frame.reshape(-1)
        scaled_seq[t, feature_dim:] = frame_vel

    return scaled_seq


def rotate_sequence(sequence, angle_range=ROTATION_ANGLE_RANGE):
    """회전 변환"""
    angle = np.random.uniform(angle_range[0], angle_range[1])
    angle_rad = angle * np.pi / 180

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    T, D = sequence.shape
    rotated_seq = np.zeros_like(sequence)

    feature_dim = D // 2

    for t in range(T):
        frame = sequence[t, :feature_dim].reshape(-1, 3)
        xy = frame[:, :2]

        rotated_xy = xy @ rotation_matrix.T
        frame_rot = frame.copy()
        frame_rot[:, :2] = rotated_xy

        rotated_seq[t, :feature_dim] = frame_rot.reshape(-1)
        rotated_seq[t, feature_dim:] = sequence[t, feature_dim:]

    return rotated_seq


def velocity_perturbation(sequence, noise_ratio=VELOCITY_PERTURB_RATIO):
    """속도 패턴에 불규칙성 추가"""
    perturbed = sequence.copy()
    n_frames = len(sequence)

    flip_indices = np.random.choice(
        n_frames,
        size=int(n_frames * noise_ratio),
        replace=False
    )

    for idx in flip_indices:
        if idx > 0:
            diff = perturbed[idx] - perturbed[idx - 1]
            perturbed[idx] = perturbed[idx - 1] - diff * 0.5

    return perturbed


def add_pauses(sequence, pause_prob=PAUSE_PROBABILITY,
               pause_strength=PAUSE_STRENGTH):
    """중간에 멈춤 추가"""
    paused = sequence.copy()

    for i in range(1, len(paused)):
        if np.random.random() < pause_prob:
            diff = paused[i] - paused[i - 1]
            paused[i] = paused[i - 1] + diff * pause_strength

    return paused


# ==================== 증강 실행 ====================
def augment_class_data(class_name, target_count, split='train'):
    """
    한 클래스의 데이터 증강

    Parameters:
    -----------
    class_name: str
        클래스 이름
    target_count: int
        목표 개수
    split: str
        'train' or 'val'

    Returns:
    --------
    augmented: np.array (target_count, 30, 336)
    """
    class_dir = os.path.join(PROCESSED_DIR, class_name)

    # ⭐ split에 따라 다른 파일 로드
    if split == 'train':
        data_path = os.path.join(class_dir, 'train_normalized.npy')
    else:
        data_path = os.path.join(class_dir, 'val_normalized.npy')

    if not os.path.exists(data_path):
        print(f"❌ [{split.upper()}] {class_name} 데이터 없음")
        return np.array([])

    # 원본 로드
    original = np.load(data_path)
    original_count = len(original)

    print(f"   [{split.upper():5s}] {class_name:12s}: {original_count:4d} → {target_count:4d} "
          f"(x{target_count / original_count:.1f})")

    # 증강 필요 개수
    n_augment_needed = target_count - original_count

    if n_augment_needed <= 0:
        if n_augment_needed < 0:
            sampled_indices = np.random.choice(original_count, target_count, replace=False)
            return original[sampled_indices]
        else:
            return original

    # 증강 방법 목록
    augmentation_methods = []

    if USE_SPEED_CHANGE:
        augmentation_methods.append(lambda seq: speed_change(seq, 0.8))
        augmentation_methods.append(lambda seq: speed_change(seq, 1.2))

    if USE_TEMPORAL_NOISE:
        augmentation_methods.append(lambda seq: add_temporal_noise(seq))

    if USE_SPATIAL_SCALE:
        augmentation_methods.append(lambda seq: scale_sequence(seq))

    if USE_ROTATION:
        augmentation_methods.append(lambda seq: rotate_sequence(seq))

    if USE_VELOCITY_PERTURB:
        augmentation_methods.append(lambda seq: velocity_perturbation(seq))

    if USE_PAUSE:
        augmentation_methods.append(lambda seq: add_pauses(seq))

    # 증강 실행
    augmented_list = [original]

    for i in tqdm(range(n_augment_needed), desc=f"     증강 ({split})", leave=False):
        sample_idx = np.random.randint(0, original_count)
        sample = original[sample_idx]

        aug_method = np.random.choice(augmentation_methods)
        augmented = aug_method(sample)

        augmented_list.append(augmented[np.newaxis, :, :])

    # 결합
    result = np.concatenate(augmented_list, axis=0)

    # 셔플
    shuffle_indices = np.random.permutation(len(result))
    result = result[shuffle_indices]

    return result


# ==================== 메인 ====================
def main():
    print("=" * 60)
    print("🔄 Step 4: 데이터 증강 (Train + Val)")
    print("=" * 60)

    print(f"\n증강 목표 (80:20 비율):")
    print(f"   [Train] 각 수화: {AUGMENT_TARGET['sign_train']}개, none: {AUGMENT_TARGET['none_train']}개")
    print(f"   [Val]   각 수화: {AUGMENT_TARGET['sign_val']}개, none: {AUGMENT_TARGET['none_val']}개")
    print(f"   총 Train: {AUGMENT_TARGET['sign_train'] * 8 + AUGMENT_TARGET['none_train']}개")
    print(f"   총 Val: {AUGMENT_TARGET['sign_val'] * 8 + AUGMENT_TARGET['none_val']}개")

    print(f"\n활성화된 증강 기법:")
    if USE_SPEED_CHANGE:
        print(f"   ✅ 속도 변화")
    if USE_TEMPORAL_NOISE:
        print(f"   ✅ 시간축 노이즈")
    if USE_SPATIAL_SCALE:
        print(f"   ✅ 공간 스케일")
    if USE_ROTATION:
        print(f"   ✅ 회전")
    if USE_VELOCITY_PERTURB:
        print(f"   ✅ 속도 섭동")
    if USE_PAUSE:
        print(f"   ✅ 멈춤 추가")

    # ===== Train 증강 =====
    print("\n[1/4] Train 데이터 증강 중...")

    X_train = []
    y_train = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        # 목표 개수
        if class_name == 'none':
            target = AUGMENT_TARGET['none_train']
        else:
            target = AUGMENT_TARGET['sign_train']

        # 증강
        augmented = augment_class_data(class_name, target, split='train')

        if len(augmented) > 0:
            X_train.append(augmented)
            y_train.extend([class_idx] * len(augmented))

    # 결합 및 셔플
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)

    shuffle_indices = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    print(f"\n✅ Train 증강 완료: {X_train.shape}")

    # ===== Val 증강 (⭐ 추가!) =====
    print("\n[2/4] Val 데이터 증강 중...")

    X_val = []
    y_val = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        # 목표 개수
        if class_name == 'none':
            target = AUGMENT_TARGET['none_val']
        else:
            target = AUGMENT_TARGET['sign_val']

        # 증강
        augmented = augment_class_data(class_name, target, split='val')

        if len(augmented) > 0:
            X_val.append(augmented)
            y_val.extend([class_idx] * len(augmented))

    # 결합 및 셔플
    if len(X_val) > 0:
        X_val = np.concatenate(X_val, axis=0)
        y_val = np.array(y_val)

        shuffle_indices = np.random.permutation(len(X_val))
        X_val = X_val[shuffle_indices]
        y_val = y_val[shuffle_indices]

        print(f"✅ Val 증강 완료: {X_val.shape}")
    else:
        X_val = np.array([])
        y_val = np.array([])
        print("⚠️  Val 데이터 없음")

    # ===== 저장 =====
    print("\n[3/4] 데이터 저장 중...")

    os.makedirs(AUGMENTED_DIR, exist_ok=True)

    np.save(os.path.join(AUGMENTED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(AUGMENTED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(AUGMENTED_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(AUGMENTED_DIR, 'y_val.npy'), y_val)

    print(f"✅ 저장 완료:")
    print(f"   - X_train.npy: {X_train.shape}")
    print(f"   - y_train.npy: {y_train.shape}")
    print(f"   - X_val.npy: {X_val.shape}")
    print(f"   - y_val.npy: {y_val.shape}")

    # ===== 통계 =====
    print("\n[4/4] 통계 수집 중...")

    train_class_counts = {}
    val_class_counts = {}

    for class_idx in range(NUM_CLASSES):
        train_class_counts[class_idx] = np.sum(y_train == class_idx)
        val_class_counts[class_idx] = np.sum(y_val == class_idx)

    # 통계 출력
    print("\n" + "=" * 60)
    print("📊 최종 통계")
    print("=" * 60)

    print("\nTrain 클래스별:")
    for class_idx in range(NUM_CLASSES):
        count = train_class_counts[class_idx]
        pct = count / len(y_train) * 100
        print(f"   {CLASS_NAMES[class_idx]:12s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nTrain 총합: {len(y_train)}개")

    if len(y_val) > 0:
        print("\nVal 클래스별:")
        for class_idx in range(NUM_CLASSES):
            count = val_class_counts[class_idx]
            pct = count / len(y_val) * 100
            print(f"   {CLASS_NAMES[class_idx]:12s}: {count:4d} ({pct:5.1f}%)")

        print(f"\nVal 총합: {len(y_val)}개")

        # 비율 계산
        ratio = len(y_train) / (len(y_train) + len(y_val)) * 100
        print(f"\n📊 Train:Val 비율 = {ratio:.1f}:{100-ratio:.1f}")

    # ===== 증강 리포트 저장 =====
    print("\n증강 리포트 생성 중...")

    # 원본 개수 수집
    original_train_counts = {}
    original_val_counts = {}

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(PROCESSED_DIR, class_name)

        # Train 원본
        train_norm_path = os.path.join(class_dir, 'train_normalized.npy')
        if os.path.exists(train_norm_path):
            data = np.load(train_norm_path)
            original_train_counts[class_name] = len(data)
        else:
            original_train_counts[class_name] = 0

        # Val 원본
        val_norm_path = os.path.join(class_dir, 'val_normalized.npy')
        if os.path.exists(val_norm_path):
            data = np.load(val_norm_path)
            original_val_counts[class_name] = len(data)
        else:
            original_val_counts[class_name] = 0

    # 리포트 작성
    report_path = os.path.join(AUGMENTED_DIR, 'augmentation_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("데이터 증강 리포트 (Train + Val)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"증강 목표 (80:20 비율)\n")
        f.write(f"  Train: 수화 {AUGMENT_TARGET['sign_train']}개/클래스, none {AUGMENT_TARGET['none_train']}개\n")
        f.write(f"  Val:   수화 {AUGMENT_TARGET['sign_val']}개/클래스, none {AUGMENT_TARGET['none_val']}개\n\n")

        f.write("=" * 60 + "\n")
        f.write("[Train 데이터]\n")
        f.write("=" * 60 + "\n\n")

        f.write("원본 → 증강:\n")
        for class_idx, class_name in enumerate(CLASS_NAMES):
            original = original_train_counts[class_name]
            augmented = train_class_counts[class_idx]
            multiplier = augmented / original if original > 0 else 0
            f.write(f"  {class_name:12s}: {original:4d} → {augmented:4d} (x{multiplier:.1f})\n")

        total_sign_train = sum([train_class_counts[i] for i in range(8)])
        total_none_train = train_class_counts[8]

        f.write(f"\n총 수화: {total_sign_train}개\n")
        f.write(f"총 none: {total_none_train}개\n")
        f.write(f"전체: {len(y_train)}개\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("[Val 데이터]\n")
        f.write("=" * 60 + "\n\n")

        f.write("원본 → 증강:\n")
        for class_idx, class_name in enumerate(CLASS_NAMES):
            original = original_val_counts[class_name]
            augmented = val_class_counts[class_idx]
            multiplier = augmented / original if original > 0 else 0
            f.write(f"  {class_name:12s}: {original:4d} → {augmented:4d} (x{multiplier:.1f})\n")

        total_sign_val = sum([val_class_counts[i] for i in range(8)])
        total_none_val = val_class_counts[8]

        f.write(f"\n총 수화: {total_sign_val}개\n")
        f.write(f"총 none: {total_none_val}개\n")
        f.write(f"전체: {len(y_val)}개\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("[전체 통계]\n")
        f.write("=" * 60 + "\n")

        total_data = len(y_train) + len(y_val)
        train_ratio = len(y_train) / total_data * 100
        val_ratio = len(y_val) / total_data * 100

        f.write(f"Train: {len(y_train)}개 ({train_ratio:.1f}%)\n")
        f.write(f"Val:   {len(y_val)}개 ({val_ratio:.1f}%)\n")
        f.write(f"총합:  {total_data}개\n")
        f.write(f"\n비율: {train_ratio:.1f}:{val_ratio:.1f} ✅\n")

    print(f"✅ 증강 리포트 저장: {report_path}")

    print("\n" + "=" * 60)
    print("✅ 데이터 증강 완료!")
    print("=" * 60)
    print(f"💾 저장 위치: {AUGMENTED_DIR}")
    print(f"📝 리포트: {report_path}")
    print(f"\n📊 Train:Val = {train_ratio:.1f}:{val_ratio:.1f} (목표 80:20) ✅")
    print("\n💡 다음 단계: 5_train_lstm.py 실행")
    print("=" * 60)


if __name__ == '__main__':
    main()