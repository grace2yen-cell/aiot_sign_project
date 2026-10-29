"""
1_extract_landmarks.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: 비디오에서 랜드마크 추출 (Holistic 기반 + 상대 좌표)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 목표:
- MediaPipe Holistic로 Hands + Face + Pose 랜드마크 추출
- ⭐ 상대 좌표 사용 (어깨 기준 정규화)
- 30프레임 (1초) 시퀀스 생성
- Face 8개 핵심 포인트만 선택
- ⭐ 28-29프레임 비디오는 자동 패딩 처리

📊 출력:
- processed_data/클래스명/train_raw.npy: (N, 30, 168)
- processed_data/클래스명/val_raw.npy: (M, 30, 168)

🔍 차원:
- Hands: 126 (양손)
- Face: 24 (8개 포인트)
- Pose: 18 (어깨, 팔꿈치, 손목)
- Total: 168
"""

import gpu_config  # GPU 설정

import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
from config import *

import warnings

warnings.filterwarnings('ignore')

# ==================== MediaPipe 초기화 ====================
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,  # 비디오 모드
    model_complexity=MP_MODEL_COMPLEXITY,
    refine_face_landmarks=MP_REFINE_FACE_LANDMARKS,
    min_detection_confidence=MP_DETECTION_CONFIDENCE,
    min_tracking_confidence=MP_TRACKING_CONFIDENCE
)

print("✅ MediaPipe Holistic 초기화 완료")
print("   ⭐ 상대 좌표 사용 (어깨 기준 정규화)")
print("   ⭐ 자동 패딩 활성화 (28-29프레임 → 30프레임)")


# ==================== 랜드마크 추출 함수 ====================
def extract_holistic_landmarks(results):
    """
    Holistic 결과 → 168차원 feature vector (상대 좌표)

    구성:
    - Hands: 126 (양손)
    - Face: 24 (선택된 8개 포인트)
    - Pose: 18 (어깨, 팔꿈치, 손목)

    ⭐ 모든 y값은 어깨 중심 기준으로 정규화!
    """

    # ---------- 1) Hands (126) ----------
    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        )

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        )

    # ---------- 2) Face (24: 8개 포인트 × 3) ----------
    face_vec = np.zeros((8, 3))

    if results.face_landmarks:
        lm = results.face_landmarks.landmark
        face_vec = np.array([
            [lm[idx].x, lm[idx].y, lm[idx].z] for idx in FACE_INDICES
        ])

    # ---------- 3) Pose (18: 6개 포인트 × 3) ----------
    pose_vec = np.zeros((6, 3))
    shoulder_center_y = 0.5  # 기본값

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # 어깨
        left_sh = lm[SHOULDER_LEFT_IDX]
        right_sh = lm[SHOULDER_RIGHT_IDX]

        # 팔꿈치
        left_elbow = lm[mp_holistic.PoseLandmark.LEFT_ELBOW]
        right_elbow = lm[mp_holistic.PoseLandmark.RIGHT_ELBOW]

        # 손목
        left_wrist = lm[mp_holistic.PoseLandmark.LEFT_WRIST]
        right_wrist = lm[mp_holistic.PoseLandmark.RIGHT_WRIST]

        # ⭐ 어깨 중심 y 좌표 계산 (기준점)
        shoulder_center_y = (left_sh.y + right_sh.y) / 2

        pose_vec = np.array([
            [left_sh.x, left_sh.y, left_sh.z],
            [right_sh.x, right_sh.y, right_sh.z],
            [left_elbow.x, left_elbow.y, left_elbow.z],
            [right_elbow.x, right_elbow.y, right_elbow.z],
            [left_wrist.x, left_wrist.y, left_wrist.z],
            [right_wrist.x, right_wrist.y, right_wrist.z],
        ])

    # ⭐⭐⭐ 핵심: 모든 y값을 어깨 중심 기준으로 정규화!
    left_hand[:, 1] -= shoulder_center_y
    right_hand[:, 1] -= shoulder_center_y
    face_vec[:, 1] -= shoulder_center_y
    pose_vec[:, 1] -= shoulder_center_y

    # ---------- 결합 ----------
    hands_vec = np.concatenate([left_hand, right_hand], axis=0).flatten()  # (126,)
    face_vec = face_vec.flatten()  # (24,)
    pose_vec = pose_vec.flatten()  # (18,)

    # ---------- 최종 feature vector ----------
    result = np.concatenate([hands_vec, face_vec, pose_vec], axis=0)  # (168,)

    return result


# ==================== 비디오 시퀀스 추출 ====================
def extract_sequence_from_video(video_path, target_frames=SEQUENCE_LENGTH):
    """
    비디오에서 균등하게 프레임 추출 → 시퀀스 생성
    ⭐ 28-29프레임인 경우 자동으로 마지막 프레임 패딩!

    Parameters:
    -----------
    video_path: str
        비디오 파일 경로
    target_frames: int
        추출할 프레임 수 (기본값: 30)

    Returns:
    --------
    sequence: np.array (30, 168) or None
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"⚠️  비디오를 열 수 없습니다: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < target_frames - 2:  # 27프레임 이하는 버림
        print(f"⚠️  프레임 수 부족 ({total_frames} < {target_frames-2}): {video_path}")
        cap.release()
        return None

    # 균등 간격 프레임 인덱스
    frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

    sequence = []
    frame_idx = 0
    next_target = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 필요한 프레임인지 확인
        if next_target < len(frame_indices) and frame_idx == frame_indices[next_target]:
            # BGR → RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 랜드마크 추출 (상대 좌표)
            results = holistic.process(rgb_frame)
            feature = extract_holistic_landmarks(results)  # (168,)

            sequence.append(feature)
            next_target += 1

            if len(sequence) >= target_frames:
                break

        frame_idx += 1

    cap.release()

    # ⭐⭐⭐ 스마트 패딩: 28-29프레임만 허용
    if len(sequence) >= target_frames - 2:  # 28, 29프레임
        padding_count = target_frames - len(sequence)
        for _ in range(padding_count):
            sequence.append(sequence[-1])
        if padding_count > 0:
            print(f"   ℹ️  패딩 {padding_count}프레임: {os.path.basename(video_path)}")

    # 정확히 target_frames개 수집했는지 확인
    if len(sequence) == target_frames:
        return np.array(sequence)  # (30, 168)
    else:
        print(f"⚠️  시퀀스 길이 부족 ({len(sequence)} < {target_frames-2}): {video_path}")
        return None


# ==================== 메인 처리 ====================
def process_all_videos():
    """전체 비디오 처리"""

    print("=" * 60)
    print("📌 Step 1: 비디오에서 랜드마크 시퀀스 추출")
    print("⭐ 상대 좌표 사용 (어깨 기준 정규화)")
    print("=" * 60)

    # Train/Val 폴더 확인
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Train 디렉토리가 없습니다: {TRAIN_DIR}")
        return

    if not os.path.exists(VAL_DIR):
        print(f"❌ Val 디렉토리가 없습니다: {VAL_DIR}")
        return

    # 클래스별 처리
    for class_name in CLASS_NAMES:
        print(f"\n{'=' * 60}")
        print(f"📁 클래스: {class_name}")
        print(f"{'=' * 60}")

        class_dir = os.path.join(PROCESSED_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # ===== Train 처리 =====
        print(f"\n[Train] 처리 중...")
        train_sequences = []

        # person1~4 순회
        for person_folder in sorted(os.listdir(TRAIN_DIR)):
            person_path = os.path.join(TRAIN_DIR, person_folder)

            if not os.path.isdir(person_path):
                continue

            # ⭐ 모든 클래스를 동일하게 처리 (None 포함!)
            class_video_dir = os.path.join(person_path, class_name)

            if not os.path.exists(class_video_dir):
                continue

            video_files = sorted([f for f in os.listdir(class_video_dir)
                                  if f.lower().endswith(('.mp4', '.avi', '.mov'))])

            if len(video_files) > 0:
                print(f"   {person_folder}/{class_name}: {len(video_files)}개 비디오")

            for video_file in video_files:
                video_path = os.path.join(class_video_dir, video_file)
                seq = extract_sequence_from_video(video_path)

                if seq is not None:
                    train_sequences.append(seq)

        # Train 저장
        if len(train_sequences) > 0:
            train_sequences = np.array(train_sequences)
            train_save_path = os.path.join(class_dir, 'train_raw.npy')
            np.save(train_save_path, train_sequences)

            print(f"\n✅ Train 저장: {train_sequences.shape}")
            print(f"   {train_save_path}")
        else:
            print(f"\n⚠️  Train 데이터 없음")

        # ===== Val 처리 =====
        print(f"\n[Val] 처리 중...")
        val_sequences = []

        # person5 처리
        person5_path = os.path.join(VAL_DIR, 'person5')

        if os.path.exists(person5_path):
            # ⭐ 모든 클래스를 동일하게 처리 (None 포함!)
            class_video_dir = os.path.join(person5_path, class_name)

            if os.path.exists(class_video_dir):
                video_files = sorted([f for f in os.listdir(class_video_dir)
                                      if f.lower().endswith(('.mp4', '.avi', '.mov'))])

                if len(video_files) > 0:
                    print(f"   person5/{class_name}: {len(video_files)}개 비디오")

                for video_file in video_files:
                    video_path = os.path.join(class_video_dir, video_file)
                    seq = extract_sequence_from_video(video_path)

                    if seq is not None:
                        val_sequences.append(seq)

        # Val 저장
        if len(val_sequences) > 0:
            val_sequences = np.array(val_sequences)
            val_save_path = os.path.join(class_dir, 'val_raw.npy')
            np.save(val_save_path, val_sequences)

            print(f"\n✅ Val 저장: {val_sequences.shape}")
            print(f"   {val_save_path}")
        else:
            print(f"\n⚠️  Val 데이터 없음")

    print("\n" + "=" * 60)
    print("✅ 랜드마크 추출 완료!")
    print("=" * 60)
    print(f"💾 저장 위치: {PROCESSED_DIR}")
    print(f"📊 각 클래스별로 train_raw.npy, val_raw.npy 생성됨")
    print(f"⭐ 28-29프레임 비디오는 자동 패딩 처리됨")
    print("=" * 60)


# ==================== 실행 ====================
if __name__ == '__main__':
    process_all_videos()