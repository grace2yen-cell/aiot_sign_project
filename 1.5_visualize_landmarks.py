"""
1.5_visualize_landmarks.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1.5: 랜드마크 시각화 (확인용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 목표:
- 추출된 랜드마크가 제대로 되었는지 시각적 확인
- 각 클래스별 샘플 이미지 생성
- 디버깅 및 보고서용

📊 입력:
- raw_videos/train/person*/클래스명/*.mp4
- raw_videos/val/person5/클래스명/*.mp4

📊 출력:
- visualization/1.5_landmarks/train/클래스명/*.png
- visualization/1.5_landmarks/val/클래스명/*.png

💡 시각화 내용:
- 양손: 빨강(왼손) / 파랑(오른손)
- 얼굴: 초록 (8개 핵심 포인트)
- Pose: 노랑 (어깨, 팔꿈치, 손목)
- 각 클래스별 5개 샘플
"""

import gpu_config  # GPU 설정

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from config import *

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=MP_MODEL_COMPLEXITY,
    refine_face_landmarks=MP_REFINE_FACE_LANDMARKS,
    min_detection_confidence=MP_DETECTION_CONFIDENCE,
    min_tracking_confidence=MP_TRACKING_CONFIDENCE
)


# ==================== 랜드마크 그리기 ====================
def draw_landmarks_on_frame(frame, results):
    """
    프레임에 Holistic 랜드마크 그리기

    색상:
    - 왼손: 빨강
    - 오른손: 파랑
    - 얼굴: 초록 (8개 포인트)
    - Pose: 노랑
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # ===== 왼손 (빨강) =====
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 100, 100), thickness=2)
        )

    # ===== 오른손 (파랑) =====
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(100, 100, 255), thickness=2)
        )

    # ===== 얼굴 (초록, 8개 포인트만) =====
    if results.face_landmarks:
        lm = results.face_landmarks.landmark

        for idx in FACE_INDICES:
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(annotated, str(idx), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # ===== Pose (노랑) =====
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

        # 좌표 변환
        points = []
        for landmark in [left_sh, right_sh, left_elbow, right_elbow,
                         left_wrist, right_wrist]:
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
            cv2.circle(annotated, (x, y), 6, (0, 255, 255), -1)

        # 연결선
        cv2.line(annotated, points[0], points[1], (0, 255, 255), 3)  # 어깨
        cv2.line(annotated, points[0], points[2], (0, 255, 255), 3)  # 왼팔
        cv2.line(annotated, points[2], points[4], (0, 255, 255), 3)
        cv2.line(annotated, points[1], points[3], (0, 255, 255), 3)  # 오른팔
        cv2.line(annotated, points[3], points[5], (0, 255, 255), 3)

    return annotated


# ==================== 비디오에서 샘플 프레임 추출 ===================
def extract_sample_frames(video_path, num_frames=5):
    """
    비디오에서 균등하게 샘플 프레임 추출 + 랜드마크 그리기

    Parameters:
    -----------
    video_path: str
    num_frames: int (기본값: 5)

    Returns:
    --------
    frames: list of np.array (RGB)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        cap.release()
        return []

    # 균등 간격
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    sampled_frames = []
    frame_idx = 0
    next_target = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if next_target < len(frame_indices) and frame_idx == frame_indices[next_target]:
            # BGR → RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 랜드마크 추출 및 그리기
            results = holistic.process(rgb_frame)
            annotated = draw_landmarks_on_frame(rgb_frame, results)

            sampled_frames.append(annotated)
            next_target += 1

            if len(sampled_frames) >= num_frames:
                break

        frame_idx += 1

    cap.release()

    return sampled_frames


# ==================== 시각화 그리드 생성 ====================
def create_visualization_grid(frames, class_name, save_path):
    """
    5개 프레임을 1행 5열 그리드로 시각화

    Parameters:
    -----------
    frames: list of np.array (5개)
    class_name: str
    save_path: str
    """
    if len(frames) == 0:
        return

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].set_title(f'Frame {i + 1}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'Class: {class_name}\n'
                 f'(Red=Left Hand, Blue=Right Hand, Green=Face, Yellow=Pose)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==================== 메인 처리 ====================
def visualize_all_classes():
    """모든 클래스 시각화"""

    print("=" * 60)
    print("🎨 Step 1.5: 랜드마크 시각화")
    print("=" * 60)

    viz_base = os.path.join(BASE_DIR, 'visualization', '1.5_landmarks')

    # Train/Val 폴더 생성
    for split in ['train', 'val']:
        for class_name in CLASS_NAMES:
            class_viz_dir = os.path.join(viz_base, split, class_name)
            os.makedirs(class_viz_dir, exist_ok=True)

    # ===== Train 시각화 =====
    print("\n[Train] 시각화 중...")

    for class_name in CLASS_NAMES:
        print(f"\n📁 {class_name}")

        class_viz_dir = os.path.join(viz_base, 'train', class_name)

        # person1~4 순회
        for person_folder in sorted(os.listdir(TRAIN_DIR)):
            person_path = os.path.join(TRAIN_DIR, person_folder)

            if not os.path.isdir(person_path):
                continue

            # ✅ 수정: None도 다른 클래스와 동일하게 처리
            # None 클래스 폴더 경로
            class_video_dir = os.path.join(person_path, class_name)

            if not os.path.exists(class_video_dir):
                continue

            # 해당 클래스 폴더 안의 비디오 파일들
            video_files = sorted([f for f in os.listdir(class_video_dir)
                                  if f.lower().endswith(('.mp4', '.avi', '.mov'))])

            # 첫 번째 비디오만 시각화
            if len(video_files) > 0:
                video_path = os.path.join(class_video_dir, video_files[0])
                frames = extract_sample_frames(video_path, num_frames=5)

                if len(frames) > 0:
                    save_path = os.path.join(class_viz_dir,
                                             f'{person_folder}_{video_files[0][:-4]}.png')
                    create_visualization_grid(frames,
                                              f'{person_folder}/{class_name}/{video_files[0]}',
                                              save_path)
                    print(f"   ✓ {person_folder}/{class_name}/{video_files[0]}")

    # ===== Val 시각화 =====
    print("\n[Val] 시각화 중...")

    person5_path = os.path.join(VAL_DIR, 'person5')

    if os.path.exists(person5_path):
        for class_name in CLASS_NAMES:
            print(f"\n📁 {class_name}")

            class_viz_dir = os.path.join(viz_base, 'val', class_name)

            # ✅ 수정: None도 다른 클래스와 동일하게 처리
            class_video_dir = os.path.join(person5_path, class_name)

            if os.path.exists(class_video_dir):
                video_files = sorted([f for f in os.listdir(class_video_dir)
                                      if f.lower().endswith(('.mp4', '.avi', '.mov'))])

                # 첫 번째 비디오만
                if len(video_files) > 0:
                    video_path = os.path.join(class_video_dir, video_files[0])
                    frames = extract_sample_frames(video_path, num_frames=5)

                    if len(frames) > 0:
                        save_path = os.path.join(class_viz_dir,
                                                 f'person5_{video_files[0][:-4]}.png')
                        create_visualization_grid(frames,
                                                  f'person5/{class_name}/{video_files[0]}',
                                                  save_path)
                        print(f"   ✓ person5/{class_name}/{video_files[0]}")

    print("\n" + "=" * 60)
    print("✅ 시각화 완료!")
    print("=" * 60)
    print(f"💾 저장 위치: {viz_base}")
    print("\n💡 확인 사항:")
    print("   1. 양손이 제대로 감지되었는가?")
    print("   2. 얼굴 8개 포인트가 올바른 위치인가?")
    print("   3. Pose (어깨, 팔꿈치, 손목)가 정확한가?")
    print("   4. 랜드마크가 흔들리거나 끊기지 않는가?")
    print("\n💡 색상 범례:")
    print("   🔴 빨강: 왼손")
    print("   🔵 파랑: 오른손")
    print("   🟢 초록: 얼굴 (8개 핵심 포인트)")
    print("   🟡 노랑: Pose (어깨, 팔꿈치, 손목)")
    print("=" * 60)


# ==================== 실행 ====================
if __name__ == '__main__':
    visualize_all_classes()

    print("\n💡 다음 단계: 2_normalize_and_add_velocity.py 실행")