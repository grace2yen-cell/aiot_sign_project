"""
server_support_with_postprocess.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
수화 인식 보조 모듈 (서버용) - 후처리 로직 포함
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⭐ 핵심 원칙:
   1. 모델 신뢰도 > 0.6 → 후처리 안 함!
   2. 후처리는 정말 애매한 경우만 (< 0.5)
   3. DOWN 동작은 절대 안 건드림

- 비디오 파일 추론 (main)
- 웹캠 실시간 추론 (run_webcam_inference)
- 후처리 로직 포함 (8_1_video_process.py 기반)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import gpu_config  # GPU 설정

import cv2
import numpy as np
from collections import deque
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
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

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
            cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)

        # 연결선
        cv2.line(frame, points[0], points[1], (0, 255, 255), 3)  # 어깨
        cv2.line(frame, points[0], points[2], (0, 255, 255), 3)  # 왼팔
        cv2.line(frame, points[2], points[4], (0, 255, 255), 3)
        cv2.line(frame, points[1], points[3], (0, 255, 255), 3)  # 오른팔
        cv2.line(frame, points[3], points[5], (0, 255, 255), 3)

    return frame


# ==================== ⭐ 후처리: 손 벌어짐 분석 ====================
def analyze_hand_spread(sequence):
    """
    손 벌어짐 분석 (8_1_video_process.py에서 가져옴)

    Parameters:
    -----------
    sequence: list or np.array
        시퀀스 버퍼 (30, 168)

    Returns:
    --------
    spread_type: str
        'wide', 'narrow', 'unknown'
    avg_distance: float
        양손 간 평균 거리
    x_variance: float
        x 좌표 분산
    """
    if len(sequence) < 5:
        return 'unknown', 0.0, 0.0

    seq = np.array(sequence)

    distances = []
    x_coords = []

    for frame in seq:
        # 왼손 첫 번째 점 (x, y)
        left_x = frame[0]
        # 오른손 첫 번째 점 (x, y)
        right_x = frame[63]
        left_y = frame[1]
        right_y = frame[64]

        # 양손 간 거리
        distance = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)
        distances.append(distance)
        x_coords.extend([left_x, right_x])

    avg_distance = np.mean(distances)
    x_variance = np.var(x_coords)

    # ⭐ threshold 높임 (더 확실한 경우만)
    if avg_distance > 0.35 or x_variance > 0.025:  # 더 높게!
        spread_type = 'wide'
    elif avg_distance < 0.12 and x_variance < 0.008:  # 더 낮게!
        spread_type = 'narrow'
    else:
        spread_type = 'unknown'

    return spread_type, avg_distance, x_variance


# ==================== 비디오 파일 추론 (서버용, 후처리 포함) ====================
def main(video_path, show_video=True):
    """
    비디오 파일에서 수화 인식 (후처리 로직 포함)

    Parameters:
    -----------
    video_path: str
        입력 비디오 파일 경로
    show_video: bool
        화면 표시 여부

    Returns:
    --------
    result: str
        최종 예측 결과 (가장 많이 나온 클래스)
    """
    print("=" * 80)
    print("🚀 비디오 파일 수화 인식 (⭐ 후처리 포함!)")
    print("=" * 80)
    print(f"📹 비디오: {video_path}\n")

    # [1] MediaPipe 초기화
    print("[1/4] MediaPipe Holistic 초기화...")
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_TRACKING_CONFIDENCE
    )
    print("✅ 초기화 완료")

    # [2] 모델 로드
    print("\n[2/4] 모델 로드...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일 없음: {MODEL_PATH}")
        holistic.close()
        return "none"

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
        return "none"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패")
        holistic.close()
        return "none"

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
    correction_count = 0  # ⭐ 후처리 보정 횟수
    prediction_history = []
    last_prediction = None
    last_confidence = 0

    # 최고 신뢰도 추적 (none 제외)
    max_confidence = 0
    max_confidence_class = None
    max_confidence_idx = None

    # [4] 추론
    print("\n[4/4] 추론 시작... (⭐ 모델 신뢰도 우선!)")
    print(f"   시퀀스 길이: {SEQUENCE_LENGTH}")
    print("=" * 80)

    # 인덱스 미리 가져오기
    idx_soundup = CLASS_MAPPING.get('soundup')
    idx_blindup = CLASS_MAPPING.get('blindup')
    idx_tempup = CLASS_MAPPING.get('tempup')

    try:
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
                    correction_reason = ""

                    # ========== ⭐ 후처리 (최소화!) ==========

                    # ⭐ 원칙 1: 신뢰도 > 0.6 → 후처리 안 함!
                    if confidence > 0.6:
                        print(f"{class_idx}_{class_name}_({confidence * 100:.1f}%) [높은 신뢰도 - 후처리 skip]")

                    # ⭐ 원칙 2: 신뢰도 낮을 때만 (< 0.5) 조심스럽게 보정
                    elif confidence < 0.5:

                        # Case 1: blindup으로 예측됐는데 soundup 특징이 강한 경우만
                        if class_name == 'blindup':

                            spread_type, avg_dist, x_var = analyze_hand_spread(sequence)

                            # 손이 **명확하게** 벌어진 경우만
                            if spread_type == 'wide':
                                if idx_soundup is not None:
                                    soundup_conf = float(pred[0][idx_soundup])

                                    # soundup 신뢰도가 blindup보다 높으면
                                    if soundup_conf > confidence:
                                        print(
                                            f"🔁 보정: blindup({confidence * 100:.1f}%) → soundup({soundup_conf * 100:.1f}%)")
                                        print(f"   이유: 손 명확히 벌어짐 (dist={avg_dist:.3f})")
                                        class_idx = idx_soundup
                                        class_name = 'soundup'
                                        confidence = soundup_conf
                                        correction_count += 1
                                        correction_reason = "blind→sound"

                        # Case 2: tempup으로 예측됐는데 soundup 특징이 강한 경우만
                        elif class_name == 'tempup':

                            spread_type, avg_dist, x_var = analyze_hand_spread(sequence)

                            if spread_type == 'wide':
                                if idx_soundup is not None:
                                    soundup_conf = float(pred[0][idx_soundup])

                                    # soundup 신뢰도가 더 높으면
                                    if soundup_conf > confidence + 0.1:  # 10% 이상 차이
                                        print(
                                            f"🔁 보정: tempup({confidence * 100:.1f}%) → soundup({soundup_conf * 100:.1f}%)")
                                        print(f"   이유: 손 명확히 벌어짐 (dist={avg_dist:.3f})")
                                        class_idx = idx_soundup
                                        class_name = 'soundup'
                                        confidence = soundup_conf
                                        correction_count += 1
                                        correction_reason = "temp→sound"

                    # 기본 출력
                    if not correction_reason:
                        print(f"{class_idx}_{class_name}_({confidence * 100:.1f}%)")

                    # 결과 저장
                    prediction_history.append((class_name, confidence))

                    # 최고 신뢰도 업데이트 (none 제외)
                    if class_name.lower() != 'none' and confidence > max_confidence:
                        max_confidence = confidence
                        max_confidence_class = class_name
                        max_confidence_idx = class_idx

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
                    print("\n⚠️  사용자에 의해 중단됨")
                    break

    except KeyboardInterrupt:
        print("\n⚠️  사용자 중단")

    finally:
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
        print(f"⭐ 후처리 보정: {correction_count}회 (최소화!)")

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
                if most_common_avg_conf >= 0.9:
                    print(f"\n😊 평균 신뢰도 {most_common_avg_conf * 100:.1f}% - 매우 정확합니다!")
            else:
                print(f"⚠️  불일치!")
                print(f"   A) 빈도 기준: {most_common_idx}_{most_common_name} "
                      f"({most_common_count}회, 평균 {most_common_avg_conf * 100:.1f}%)")
                print(f"   B) 최고 신뢰도: {max_confidence_idx}_{max_confidence_class} "
                      f"(최고 {max_confidence * 100:.1f}%)")
                print(f"\n   💡 권장: 빈도가 더 높은 '{most_common_name}'을(를) 최종 결과로 선택")

                # 😊 스마일 이모티콘 조건 체크
                if most_common_avg_conf >= 0.9:
                    print(f"\n😊 평균 신뢰도 {most_common_avg_conf * 100:.1f}% - 매우 정확합니다!")

            print("=" * 80)

            return most_common_name
        else:
            print(f"\n🎯 최종 결과: 'none' 이외의 클래스가 감지되지 않았습니다.")
            print("=" * 80)
            return "none"


# ==================== 웹캠 실시간 인식 (후처리 포함) ====================
def run_webcam_inference(camera_id=0, show_video=True, max_frames=None):
    """
    웹캠 실시간 수화 인식 (후처리 로직 포함)

    Parameters:
    -----------
    camera_id: int
        웹캠 장치 번호 (기본값: 0)
    show_video: bool
        화면 표시 여부
    max_frames: int or None
        최대 프레임 수 (None이면 무제한)

    Returns:
    --------
    result: str
        최종 예측 결과 (가장 많이 나온 클래스)
    """
    print("=" * 80)
    print("🚀 웹캠 실시간 수화 인식 (⭐ 후처리 포함!)")
    print("=" * 80)

    # MediaPipe 초기화
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_TRACKING_CONFIDENCE
    )

    # 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일 없음: {MODEL_PATH}")
        holistic.close()
        return "none"

    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        model = load_model(MODEL_PATH)

    # 정규화 파라미터 로드
    norm_params_path = os.path.join(AUGMENTED_DIR, 'normalization_params.npz')
    use_normalization = False

    if os.path.exists(norm_params_path):
        norm_data = np.load(norm_params_path)
        norm_mean = norm_data['mean']
        norm_std = norm_data['std']
        use_normalization = True

    # 웹캠 초기화
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ 웹캠 열기 실패!")
        holistic.close()
        return "none"

    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 시퀀스 버퍼
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # 통계
    frame_count = 0
    correction_count = 0
    prediction_history = []
    last_prediction = None
    last_confidence = 0

    # 최고 신뢰도 추적 (none 제외)
    max_confidence = 0
    max_confidence_class = None
    max_confidence_idx = None

    # 인덱스
    idx_soundup = CLASS_MAPPING.get('soundup')
    idx_blindup = CLASS_MAPPING.get('blindup')
    idx_tempup = CLASS_MAPPING.get('tempup')

    try:
        print("⭐ 웹캠 추론 시작! (q를 눌러 종료)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 최대 프레임 체크
            if max_frames and frame_count > max_frames:
                break

            # BGR → RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            # 랜드마크 추출
            landmarks = extract_holistic_landmarks(results)

            if landmarks is not None:
                sequence_buffer.append(landmarks)

                # 예측
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    sequence = np.array(list(sequence_buffer))

                    # Velocity 추가
                    sequence_with_velocity = add_velocity_to_sequence(sequence)

                    # 정규화
                    seq_for_model = sequence_with_velocity.copy()
                    if use_normalization:
                        mean_squeezed = norm_mean.squeeze()
                        std_squeezed = norm_std.squeeze()
                        seq_for_model = (seq_for_model - mean_squeezed) / std_squeezed

                    seq_batch = seq_for_model[np.newaxis, :, :]

                    # 예측
                    with tf.device('/GPU:0' if gpus else '/CPU:0'):
                        pred = model.predict(seq_batch, verbose=0)

                    class_idx = int(np.argmax(pred))
                    confidence = float(pred[0][class_idx])
                    class_name = CLASS_NAMES[class_idx]

                    # ========== ⭐ 후처리 로직 ==========

                    # 신뢰도 > 0.6: 후처리 안 함
                    if confidence > 0.85:
                        pass  # 그대로 사용

                    # 신뢰도 < 0.5: 조심스럽게 보정
                    elif confidence < 0.84:

                        # blindup → soundup 보정
                        if class_name == 'blindup':
                            spread_type, avg_dist, x_var = analyze_hand_spread(sequence)

                            if spread_type == 'wide' and idx_soundup is not None:
                                soundup_conf = float(pred[0][idx_soundup])

                                if soundup_conf > confidence:
                                    class_idx = idx_soundup
                                    class_name = 'soundup'
                                    confidence = soundup_conf
                                    correction_count += 1

                        # tempup → soundup 보정
                        elif class_name == 'tempup':
                            spread_type, avg_dist, x_var = analyze_hand_spread(sequence)

                            if spread_type == 'wide' and idx_soundup is not None:
                                soundup_conf = float(pred[0][idx_soundup])

                                if soundup_conf > confidence + 0.1:
                                    class_idx = idx_soundup
                                    class_name = 'soundup'
                                    confidence = soundup_conf
                                    correction_count += 1

                    # 결과 저장
                    prediction_history.append((class_name, confidence))

                    # 최고 신뢰도 업데이트 (none 제외)
                    if class_name.lower() != 'none' and confidence > max_confidence:
                        max_confidence = confidence
                        max_confidence_class = class_name
                        max_confidence_idx = class_idx

                    last_prediction = class_name
                    last_confidence = confidence

            # 화면 표시
            if show_video:
                display_frame = draw_landmarks_on_frame(frame.copy(), results)

                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if last_prediction:
                    pred_text = f"{last_prediction} ({last_confidence * 100:.1f}%)"

                    # 신뢰도에 따른 색상 변경
                    if last_confidence >= 0.8:
                        color = (0, 255, 0)  # 초록
                    elif last_confidence >= 0.6:
                        color = (0, 255, 255)  # 노란
                    else:
                        color = (0, 165, 255)  # 주황

                    cv2.putText(display_frame, pred_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                buffer_status = f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
                cv2.putText(display_frame, buffer_status, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                cv2.imshow('Webcam Sign Language Recognition', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n⚠️  사용자 중단")

    finally:
        cap.release()
        holistic.close()
        if show_video:
            cv2.destroyAllWindows()

        # 최종 결과 계산
        print("\n" + "=" * 80)
        print("📊 웹캠 추론 통계")
        print("=" * 80)
        print(f"총 프레임: {frame_count}")
        print(f"예측 수행: {len(prediction_history)}")
        print(f"⭐ 후처리 보정: {correction_count}회")

        filtered_predictions = [(name, conf) for name, conf in prediction_history
                                if name.lower() != 'none']

        if filtered_predictions:
            # 클래스별 통계
            class_stats = {}
            for class_name, confidence in filtered_predictions:
                if class_name not in class_stats:
                    class_stats[class_name] = {'count': 0, 'confidences': []}
                class_stats[class_name]['count'] += 1
                class_stats[class_name]['confidences'].append(confidence)

            # 가장 많이 나온 클래스
            most_common = max(class_stats.items(), key=lambda x: x[1]['count'])
            result = most_common[0]
            result_count = most_common[1]['count']
            result_avg_conf = np.mean(most_common[1]['confidences'])
            result_idx = CLASS_MAPPING.get(result)

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
            print(f"   예측 클래스: {result}")
            print(f"   클래스 인덱스: {result_idx}")
            print(f"   출현 횟수: {result_count}회 ({result_count / len(filtered_predictions) * 100:.1f}%)")
            print(f"   평균 신뢰도: {result_avg_conf * 100:.2f}%")

            if max_confidence_class:
                print("\n[2] 가장 높은 신뢰도:")
                print("-" * 80)
                print(f"   예측 클래스: {max_confidence_class}")
                print(f"   클래스 인덱스: {max_confidence_idx}")
                print(f"   최고 신뢰도: {max_confidence * 100:.2f}%")

            print("\n[3] 최종 판단:")
            print("=" * 80)

            if result == max_confidence_class:
                print(f"✅ 일치! 예측 결과가 일관성 있습니다.")
                print(f"   결론: {result_idx}_{result}")

                # 😊 스마일 이모티콘 조건 체크
                if result_avg_conf >= 0.9:
                    print(f"\n😊 평균 신뢰도 {result_avg_conf * 100:.1f}% - 매우 정확합니다!")
            else:
                print(f"⚠️  불일치!")
                print(f"   A) 빈도 기준: {result_idx}_{result} "
                      f"({result_count}회, 평균 {result_avg_conf * 100:.1f}%)")
                print(f"   B) 최고 신뢰도: {max_confidence_idx}_{max_confidence_class} "
                      f"(최고 {max_confidence * 100:.1f}%)")
                print(f"\n   💡 권장: 빈도가 더 높은 '{result}'을(를) 최종 결과로 선택")

                # 😊 스마일 이모티콘 조건 체크
                if result_avg_conf >= 0.9:
                    print(f"\n😊 평균 신뢰도 {result_avg_conf * 100:.1f}% - 매우 정확합니다!")

            print("=" * 80)
            return result
        else:
            print(f"\n🎯 최종 결과: 'none' 이외의 클래스가 감지되지 않았습니다.")
            print("=" * 80)
            return "none"


if __name__ == '__main__':
    # 독립 실행 시: 웹캠 실시간 인식
    result = run_webcam_inference(camera_id=0, show_video=True)
    print(f"\n최종 예측: {result}")