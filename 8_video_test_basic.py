"""
8_video_test_basic.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
비디오 기반 수화 인식 추론
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 목표:
- 비디오 파일에서 MediaPipe Holistic 랜드마크 추출
- 상대 좌표 + Velocity 피처로 LSTM 모델 추론
- 프레임별 예측 결과 수집 및 최종 결과 도출

📊 입력:
- 비디오 파일 (.mp4, .avi 등)

📊 출력:
- 프레임별 예측 결과
- 최종 예측 클래스 (빈도 + 신뢰도 기반)
"""

import gpu_config  # GPU 설정

import cv2
import numpy as np
from collections import deque, Counter
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import os
from config import *

# GPU 메모리 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 사용: {len(gpus)}개")
        for i, gpu in enumerate(gpus):
            print(f"   - GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️  GPU 없음 - CPU 사용")

# MediaPipe 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ==================== 랜드마크 추출 ====================
def extract_holistic_landmarks(results):
    """
    Holistic 랜드마크 추출 (상대 좌표)

    Returns: (168,) 벡터 or None
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

    # ---------- 2) Face (24: 8개 포인트) ----------
    face_vec = np.zeros((8, 3))

    if results.face_landmarks:
        lm = results.face_landmarks.landmark
        face_vec = np.array([
            [lm[idx].x, lm[idx].y, lm[idx].z] for idx in FACE_INDICES
        ])

    # ---------- 3) Pose (18: 6개 포인트) ----------
    pose_vec = np.zeros((6, 3))
    shoulder_center_y = 0.5  # 기본값

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        left_sh = lm[SHOULDER_LEFT_IDX]
        right_sh = lm[SHOULDER_RIGHT_IDX]
        left_elbow = lm[mp_holistic.PoseLandmark.LEFT_ELBOW]
        right_elbow = lm[mp_holistic.PoseLandmark.RIGHT_ELBOW]
        left_wrist = lm[mp_holistic.PoseLandmark.LEFT_WRIST]
        right_wrist = lm[mp_holistic.PoseLandmark.RIGHT_WRIST]

        # ⭐ 어깨 중심 계산
        shoulder_center_y = (left_sh.y + right_sh.y) / 2

        pose_vec = np.array([
            [left_sh.x, left_sh.y, left_sh.z],
            [right_sh.x, right_sh.y, right_sh.z],
            [left_elbow.x, left_elbow.y, left_elbow.z],
            [right_elbow.x, right_elbow.y, right_elbow.z],
            [left_wrist.x, left_wrist.y, left_wrist.z],
            [right_wrist.x, right_wrist.y, right_wrist.z],
        ])

    # ⭐⭐⭐ 상대 좌표 변환 (어깨 기준)
    left_hand[:, 1] -= shoulder_center_y
    right_hand[:, 1] -= shoulder_center_y
    face_vec[:, 1] -= shoulder_center_y
    pose_vec[:, 1] -= shoulder_center_y

    # ---------- 결합 ----------
    hands_vec = np.concatenate([left_hand, right_hand], axis=0).flatten()  # (126,)
    face_vec = face_vec.flatten()  # (24,)
    pose_vec = pose_vec.flatten()  # (18,)

    result = np.concatenate([hands_vec, face_vec, pose_vec], axis=0)  # (168,)

    # 손이 하나라도 감지되었는지 확인
    has_any_hand = (results.left_hand_landmarks is not None or
                    results.right_hand_landmarks is not None)

    if not has_any_hand:
        return None

    return result


# ==================== Velocity 피처 추가 ====================
def add_velocity_to_sequence(sequence):
    """
    시퀀스에 velocity 피처 추가

    Parameters:
    -----------
    sequence: np.array (30, 168)

    Returns:
    --------
    enhanced: np.array (30, 336)
    """
    # 프레임 간 차분
    velocity = np.diff(sequence, axis=0)  # (29, 168)

    # 첫 프레임은 0
    first_velocity = np.zeros((1, 168))
    velocity = np.vstack([first_velocity, velocity])  # (30, 168)

    # 원본 + 속도
    enhanced = np.concatenate([sequence, velocity], axis=1)  # (30, 336)

    return enhanced


# ==================== 시각화 ====================
def draw_landmarks_on_frame(frame, results):
    """프레임에 랜드마크 그리기"""
    # 왼손 (빨간색)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
        )

    # 오른손 (파란색)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=2)
        )

    # 얼굴 (초록색 점)
    if results.face_landmarks:
        lm = results.face_landmarks.landmark
        h, w = frame.shape[:2]

        for idx in FACE_INDICES:
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Pose (노란색)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        points = []
        for landmark_name in ['LEFT_SHOULDER', 'RIGHT_SHOULDER',
                              'LEFT_ELBOW', 'RIGHT_ELBOW',
                              'LEFT_WRIST', 'RIGHT_WRIST']:
            landmark = lm[mp_holistic.PoseLandmark[landmark_name]]
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        # 연결선
        cv2.line(frame, points[0], points[1], (0, 255, 255), 2)  # 어깨
        cv2.line(frame, points[0], points[2], (0, 255, 255), 2)  # 왼팔
        cv2.line(frame, points[2], points[4], (0, 255, 255), 2)
        cv2.line(frame, points[1], points[3], (0, 255, 255), 2)  # 오른팔
        cv2.line(frame, points[3], points[5], (0, 255, 255), 2)

    return frame


# ==================== 메인 추론 ====================
def main(video_path, show_video=True):
    """
    비디오에서 수화 인식

    Parameters:
    -----------
    video_path: str
        입력 비디오 경로
    show_video: bool
        화면 표시 여부
    """
    print("=" * 80)
    print("🚀 비디오 기반 수화 인식")
    print("=" * 80)
    print(f"📹 비디오: {video_path}\n")

    # [1] MediaPipe 초기화
    print("[1/4] MediaPipe Holistic 초기화...")
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=MP_MODEL_COMPLEXITY,
        refine_face_landmarks=MP_REFINE_FACE_LANDMARKS,
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_TRACKING_CONFIDENCE
    )
    print("✅ 초기화 완료")

    # [2] 모델 로드
    print("\n[2/4] 모델 로드...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일 없음: {MODEL_PATH}")
        holistic.close()
        return

    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        model = load_model(MODEL_PATH)
        print(f"✅ 모델 로드 완료")

    # 정규화 파라미터 로드
    norm_params_path = os.path.join(AUGMENTED_DIR, 'normalization_params.npz')
    use_normalization = False

    if os.path.exists(norm_params_path):
        norm_data = np.load(norm_params_path)
        norm_mean = norm_data['mean']
        norm_std = norm_data['std']
        use_normalization = True
        print(f"✅ 정규화 파라미터 로드")
    else:
        print(f"⚠️  정규화 파라미터 없음")

    # [3] 비디오 로드
    print("\n[3/4] 비디오 로드...")
    if not os.path.exists(video_path):
        print(f"❌ 비디오 파일 없음: {video_path}")
        holistic.close()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패")
        holistic.close()
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ 비디오 로드 완료")
    print(f"   FPS: {fps}, 총 프레임: {total_frames}")

    # 시퀀스 버퍼
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # 통계
    frame_count = 0
    detection_count = 0
    prediction_count = 0
    prediction_history = []

    # 최고 신뢰도 추적 (none 제외)
    max_confidence = 0
    max_confidence_class = None
    max_confidence_idx = None

    last_prediction = None
    last_confidence = 0

    # [4] 추론
    print("\n[4/4] 추론 시작...")
    print(f"   시퀀스 길이: {SEQUENCE_LENGTH}")
    print(f"   신뢰도 임계값: {CONFIDENCE_THRESHOLD}")
    print("=" * 80)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # 랜드마크 추출
        landmarks = extract_holistic_landmarks(results)

        if landmarks is not None:
            detection_count += 1
            sequence_buffer.append(landmarks)

            # 시퀀스 가득 참
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # 원본 시퀀스
                sequence = np.array(list(sequence_buffer))  # (30, 168)

                # Velocity 추가
                sequence_with_velocity = add_velocity_to_sequence(sequence)  # (30, 336)

                # 정규화
                seq_for_model = sequence_with_velocity.copy()
                if use_normalization:
                    mean_squeezed = norm_mean.squeeze()
                    std_squeezed = norm_std.squeeze()
                    seq_for_model = (seq_for_model - mean_squeezed) / std_squeezed

                # 배치 차원 추가
                seq_batch = seq_for_model[np.newaxis, :, :]  # (1, 30, 336)

                # 예측
                with tf.device('/GPU:0' if gpus else '/CPU:0'):
                    pred = model.predict(seq_batch, verbose=0)

                class_idx = int(np.argmax(pred))
                confidence = float(pred[0][class_idx])
                class_name = CLASS_NAMES[class_idx]

                prediction_count += 1

                # 예측 결과 저장
                prediction_history.append((class_name, confidence))

                # 최고 신뢰도 업데이트 (none 제외)
                if class_name.lower() != 'none' and confidence > max_confidence:
                    max_confidence = confidence
                    max_confidence_class = class_name
                    max_confidence_idx = class_idx

                print(f"{class_idx}_{class_name}_({confidence * 100:.1f}%)")
                last_prediction = class_name
                last_confidence = confidence

        # 화면 표시
        if show_video:
            display_frame = draw_landmarks_on_frame(frame.copy(), results)

            info_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(display_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if last_prediction:
                pred_text = f"{class_idx}_{last_prediction}_({last_confidence * 100:.1f}%)"
                cv2.putText(display_frame, pred_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Sign Language Recognition', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  사용자 중단")
                break

    # 정리
    cap.release()
    holistic.close()
    if show_video:
        cv2.destroyAllWindows()

    # 통계 분석
    print("\n" + "=" * 80)
    print("📊 추론 통계")
    print("=" * 80)
    print(f"총 프레임: {frame_count}")
    print(f"손 감지: {detection_count} ({detection_count / frame_count * 100:.1f}%)")
    print(f"예측 수행: {prediction_count}")

    # none 제외 필터링
    filtered_predictions = [(name, conf) for name, conf in prediction_history
                            if name.lower() != 'none']

    if filtered_predictions:
        # 클래스별 통계
        class_stats = {}
        for class_name, confidence in filtered_predictions:
            if class_name not in class_stats:
                class_stats[class_name] = {
                    'count': 0,
                    'confidences': []
                }
            class_stats[class_name]['count'] += 1
            class_stats[class_name]['confidences'].append(confidence)

        # 가장 많이 나온 클래스
        most_common_class = max(class_stats.items(), key=lambda x: x[1]['count'])
        most_common_name = most_common_class[0]
        most_common_count = most_common_class[1]['count']
        most_common_avg_conf = np.mean(most_common_class[1]['confidences'])
        most_common_idx = CLASS_MAPPING.get(most_common_name)

        # 클래스별 출현 통계
        print("\n" + "-" * 80)
        print("📈 클래스별 출현 통계 (none 제외)")
        print("-" * 80)

        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True)

        for class_name, stats in sorted_classes:
            class_idx = CLASS_MAPPING.get(class_name)
            avg_conf = np.mean(stats['confidences'])
            max_conf = max(stats['confidences'])
            min_conf = min(stats['confidences'])

            print(f"  {class_idx}_{class_name:12s}: {stats['count']:3d}회 "
                  f"(평균: {avg_conf * 100:5.1f}%, 최고: {max_conf * 100:5.1f}%, "
                  f"최저: {min_conf * 100:5.1f}%)")

        # 최종 결과
        print("\n" + "=" * 80)
        print("🎯 최종 결과")
        print("=" * 80)

        print("\n[1] 가장 많이 출현한 클래스:")
        print("-" * 80)
        print(f"   예측 클래스: {most_common_name}")
        print(f"   클래스 인덱스: {most_common_idx}")
        print(f"   출현 횟수: {most_common_count}회 ({most_common_count / len(filtered_predictions) * 100:.1f}%)")
        print(f"   평균 신뢰도: {most_common_avg_conf * 100:.2f}%")

        if max_confidence_class:
            print("\n[2] 가장 높은 신뢰도:")
            print("-" * 80)
            print(f"   예측 클래스: {max_confidence_class}")
            print(f"   클래스 인덱스: {max_confidence_idx}")
            print(f"   최고 신뢰도: {max_confidence * 100:.2f}%")

        print("\n[3] 최종 판단:")
        print("=" * 80)

        if most_common_name == max_confidence_class:
            print(f"✅ 일치! 예측 결과가 일관성 있습니다.")
            print(f"   결론: {most_common_idx}_{most_common_name}")

            # 😊 스마일 이모티콘 조건 체크
            if most_common_avg_conf >= 0.9:  # 평균 신뢰도 90% 이상
                print(f"\n😊 평균 신뢰도 {most_common_avg_conf * 100:.1f}% - 매우 정확합니다!")
        else:
            print(f"⚠️  불일치!")
            print(f"   A) 빈도 기준: {most_common_idx}_{most_common_name} "
                  f"({most_common_count}회, 평균 {most_common_avg_conf * 100:.1f}%)")
            print(f"   B) 최고 신뢰도: {max_confidence_idx}_{max_confidence_class} "
                  f"(최고 {max_confidence * 100:.1f}%)")
            print(f"\n   💡 권장: 빈도가 더 높은 '{most_common_name}'을(를) 최종 결과로 선택")

        print("=" * 80)

        return most_common_name
    else:
        print("\n🎯 최종 결과: 'none' 이외의 클래스가 감지되지 않았습니다.")
        return 'none'


if __name__ == '__main__':
    # ========================================
    # 🎬 비디오 경로 설정
    # ========================================
    VIDEO_PATH = "./data/raw_videos/test/lighton.mp4"  # ← 여기를 수정하세요!
    # ========================================

    SHOW_VIDEO = True  # False로 하면 화면 안 보임

    print(f"\n🎯 설정된 비디오: {VIDEO_PATH}\n")

    result = main(video_path=VIDEO_PATH, show_video=SHOW_VIDEO)

    if result:
        print(f"\n✅ 추론 완료! 최종 결과: {result}")