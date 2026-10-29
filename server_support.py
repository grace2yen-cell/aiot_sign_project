"""
server_support.py
[실험용] 후처리를 넣기 전의 서버 추론 모듈.
현재 서버(7_server.py)는 server_support_with_postprocess.py를 사용함.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
수화 인식 보조 모듈 (서버용)
- 비디오 파일 추론 (main)
- 웹캠 실시간 추론 (run_webcam_inference)
- 후처리 로직 제외 (기본 예측만 수행)
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
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU 사용: {len(gpus)}개")
    except RuntimeError as e:
        print(e)
else:
    print("GPU 없음 - CPU 사용")

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
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])

    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

    # ---------- 2) Face (24: 8개 포인트) ----------
    face_vec = np.zeros((8, 3))

    if results.face_landmarks:
        lm = results.face_landmarks.landmark
        face_vec = np.array([[lm[idx].x, lm[idx].y, lm[idx].z] for idx in FACE_INDICES])

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

        # 어깨 중심 계산
        shoulder_center_y = (left_sh.y + right_sh.y) / 2

        pose_vec = np.array(
            [
                [left_sh.x, left_sh.y, left_sh.z],
                [right_sh.x, right_sh.y, right_sh.z],
                [left_elbow.x, left_elbow.y, left_elbow.z],
                [right_elbow.x, right_elbow.y, right_elbow.z],
                [left_wrist.x, left_wrist.y, left_wrist.z],
                [right_wrist.x, right_wrist.y, right_wrist.z],
            ]
        )

    # 상대 좌표 변환 (어깨 기준)
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
    has_any_hand = (
        results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
    )

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
            mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2),
        )

    # 오른손 (파란색)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=2),
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
        for landmark_name in [
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_WRIST",
            "RIGHT_WRIST",
        ]:
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


# ==================== 비디오 파일 추론 (서버용) ====================
def main(video_path, show_video=True):
    """
    비디오 파일에서 수화 인식

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
    print("비디오 파일 수화 인식")
    print("=" * 80)
    print(f"비디오: {video_path}\n")

    # [1] MediaPipe 초기화
    print("[1/4] MediaPipe Holistic 초기화...")
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=MP_MODEL_COMPLEXITY,
        refine_face_landmarks=MP_REFINE_FACE_LANDMARKS,
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_TRACKING_CONFIDENCE,
    )
    print("초기화 완료")

    # [2] 모델 로드
    print("\n[2/4] 모델 로드...")
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일 없음: {MODEL_PATH}")
        holistic.close()
        return "none"

    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        model = load_model(MODEL_PATH)
        print("모델 로드 완료")

    # 정규화 파라미터 로드
    norm_params_path = os.path.join(AUGMENTED_DIR, "normalization_params.npz")
    use_normalization = False

    if os.path.exists(norm_params_path):
        norm_data = np.load(norm_params_path)
        norm_mean = norm_data["mean"]
        norm_std = norm_data["std"]
        use_normalization = True
        print("정규화 파라미터 로드")
    else:
        print("정규화 파라미터 없음")

    # [3] 비디오 로드
    print("\n[3/4] 비디오 로드...")
    if not os.path.exists(video_path):
        print(f"비디오 파일 없음: {video_path}")
        holistic.close()
        return "none"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 열기 실패")
        holistic.close()
        return "none"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("비디오 로드 완료")
    print(f"   FPS: {fps}, 총 프레임: {total_frames}")

    # 시퀀스 버퍼
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # 통계
    frame_count = 0
    prediction_history = []
    last_prediction = None
    last_confidence = 0

    # [4] 추론
    print("\n[4/4] 추론 시작...")
    print(f"   시퀀스 길이: {SEQUENCE_LENGTH}")
    print("=" * 80)

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

                    # 예측 (후처리 없이 기본 예측만)
                    with tf.device("/GPU:0" if gpus else "/CPU:0"):
                        pred = model.predict(seq_batch, verbose=0)

                    class_idx = int(np.argmax(pred))
                    confidence = float(pred[0][class_idx])
                    class_name = CLASS_NAMES[class_idx]

                    # 결과 저장
                    prediction_history.append((class_name, confidence))
                    last_prediction = class_name
                    last_confidence = confidence

                    print(f"{class_idx}_{class_name}_({confidence * 100:.1f}%)")

            # 화면 표시
            if show_video:
                display_frame = draw_landmarks_on_frame(frame.copy(), results)

                info_text = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(
                    display_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if last_prediction:
                    pred_text = f"{last_prediction} ({last_confidence * 100:.1f}%)"
                    cv2.putText(
                        display_frame,
                        pred_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Sign Language Recognition", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n사용자 중단")

    finally:
        # 정리
        cap.release()
        holistic.close()
        if show_video:
            cv2.destroyAllWindows()

        # 최종 결과 계산
        print("\n" + "=" * 80)
        print("추론 통계")
        print("=" * 80)
        print(f"총 프레임: {frame_count}")
        print(f"예측 수행: {len(prediction_history)}")

        # none 제외 필터링
        filtered_predictions = [
            (name, conf) for name, conf in prediction_history if name.lower() != "none"
        ]

        if filtered_predictions:
            # 클래스별 통계
            class_stats = {}
            for class_name, confidence in filtered_predictions:
                if class_name not in class_stats:
                    class_stats[class_name] = {"count": 0, "confidences": []}
                class_stats[class_name]["count"] += 1
                class_stats[class_name]["confidences"].append(confidence)

            # 가장 많이 나온 클래스
            most_common = max(class_stats.items(), key=lambda x: x[1]["count"])
            result = most_common[0]
            result_count = most_common[1]["count"]
            result_avg_conf = np.mean(most_common[1]["confidences"])

            print(f"\n최종 결과: {result}")
            print(f"   출현 횟수: {result_count}회")
            print(f"   평균 신뢰도: {result_avg_conf * 100:.2f}%")

            # 스마일 이모티콘 조건 체크
            if result_avg_conf >= 0.9:
                print(f"\n평균 신뢰도 {result_avg_conf * 100:.1f}% - 매우 정확합니다!")

            print("=" * 80)
            return result
        else:
            print("\n인식된 수화 없음")
            print("=" * 80)
            return "none"


# ==================== 웹캠 실시간 인식 ====================
def run_webcam_inference(camera_id=0, show_video=True, max_frames=None):
    """
    웹캠 실시간 수화 인식

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
    print("웹캠 실시간 수화 인식")
    print("=" * 80)

    # MediaPipe 초기화
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=MP_MODEL_COMPLEXITY,
        refine_face_landmarks=MP_REFINE_FACE_LANDMARKS,
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_TRACKING_CONFIDENCE,
    )

    # 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일 없음: {MODEL_PATH}")
        holistic.close()
        return "none"

    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        model = load_model(MODEL_PATH)

    # 정규화 파라미터 로드
    norm_params_path = os.path.join(AUGMENTED_DIR, "normalization_params.npz")
    use_normalization = False

    if os.path.exists(norm_params_path):
        norm_data = np.load(norm_params_path)
        norm_mean = norm_data["mean"]
        norm_std = norm_data["std"]
        use_normalization = True

    # 웹캠 초기화
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("웹캠 열기 실패!")
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
    prediction_history = []
    last_prediction = None
    last_confidence = 0

    try:
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
                    with tf.device("/GPU:0" if gpus else "/CPU:0"):
                        pred = model.predict(seq_batch, verbose=0)

                    class_idx = int(np.argmax(pred))
                    confidence = float(pred[0][class_idx])
                    class_name = CLASS_NAMES[class_idx]

                    # 결과 저장
                    prediction_history.append((class_name, confidence))
                    last_prediction = class_name
                    last_confidence = confidence

                    print(f"{class_idx}_{class_name}_({confidence * 100:.1f}%)")

            # 화면 표시
            if show_video:
                display_frame = draw_landmarks_on_frame(frame.copy(), results)

                cv2.putText(
                    display_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if last_prediction:
                    pred_text = f"{last_prediction} ({last_confidence * 100:.1f}%)"

                    # 신뢰도에 따른 색상 변경
                    if last_confidence >= 0.8:
                        color = (0, 255, 0)  # 초록
                    elif last_confidence >= 0.6:
                        color = (0, 255, 255)  # 노란
                    else:
                        color = (0, 165, 255)  # 주황

                    cv2.putText(
                        display_frame, pred_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                    )

                buffer_status = f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
                cv2.putText(
                    display_frame,
                    buffer_status,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                )

                cv2.imshow("Webcam Sign Language Recognition", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n사용자 중단")

    finally:
        cap.release()
        holistic.close()
        if show_video:
            cv2.destroyAllWindows()

        # 최종 결과 계산
        filtered_predictions = [
            (name, conf) for name, conf in prediction_history if name.lower() != "none"
        ]

        if filtered_predictions:
            class_stats = {}
            for class_name, confidence in filtered_predictions:
                if class_name not in class_stats:
                    class_stats[class_name] = {"count": 0, "confidences": []}
                class_stats[class_name]["count"] += 1
                class_stats[class_name]["confidences"].append(confidence)

            most_common = max(class_stats.items(), key=lambda x: x[1]["count"])
            result = most_common[0]
            result_avg_conf = np.mean(most_common[1]["confidences"])

            print(f"\n최종 결과: {result}")
            print(f"   평균 신뢰도: {result_avg_conf * 100:.2f}%")

            # 스마일 이모티콘 조건 체크
            if result_avg_conf >= 0.9:
                print(f"\n평균 신뢰도 {result_avg_conf * 100:.1f}% - 매우 정확합니다!")

            return result
        else:
            print("\n인식된 수화 없음")
            return "none"


if __name__ == "__main__":
    # 독립 실행 시: 웹캠 실시간 인식
    result = run_webcam_inference(camera_id=0, show_video=True)
    print(f"\n최종 예측: {result}")
