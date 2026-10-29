"""
8_1_video_process.py
[실험용] sounddown -> soundup 오인식을 줄이려고 만든 후처리 실험
(검지 방향, 어깨 위치, 손 이동 방향 규칙). 서버에는 적용되지 않음.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
비디오 기반 수화 인식 + soundup/sounddown 후처리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

후처리 로직:
1. sounddown 예측 + 검지 위로 + 어깨 위 → soundup 변경
2. sounddown 예측 + 검지 아래로 + 아래로 이동 → sounddown 유지
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
    """Holistic 랜드마크 추출 (상대 좌표)"""
    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))

    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])

    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

    face_vec = np.zeros((8, 3))
    if results.face_landmarks:
        lm = results.face_landmarks.landmark
        face_vec = np.array([[lm[idx].x, lm[idx].y, lm[idx].z] for idx in FACE_INDICES])

    pose_vec = np.zeros((6, 3))
    shoulder_center_y = 0.5

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        left_sh = lm[SHOULDER_LEFT_IDX]
        right_sh = lm[SHOULDER_RIGHT_IDX]
        left_elbow = lm[mp_holistic.PoseLandmark.LEFT_ELBOW]
        right_elbow = lm[mp_holistic.PoseLandmark.RIGHT_ELBOW]
        left_wrist = lm[mp_holistic.PoseLandmark.LEFT_WRIST]
        right_wrist = lm[mp_holistic.PoseLandmark.RIGHT_WRIST]

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

    left_hand[:, 1] -= shoulder_center_y
    right_hand[:, 1] -= shoulder_center_y
    face_vec[:, 1] -= shoulder_center_y
    pose_vec[:, 1] -= shoulder_center_y

    hands_vec = np.concatenate([left_hand, right_hand], axis=0).flatten()
    face_vec = face_vec.flatten()
    pose_vec = pose_vec.flatten()

    result = np.concatenate([hands_vec, face_vec, pose_vec], axis=0)

    has_any_hand = (
        results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
    )

    if not has_any_hand:
        return None

    return result


def add_velocity_to_sequence(sequence):
    """속도 피처 추가"""
    velocity = np.diff(sequence, axis=0)
    first_velocity = np.zeros((1, 168))
    velocity = np.vstack([first_velocity, velocity])
    enhanced = np.concatenate([sequence, velocity], axis=1)
    return enhanced


# ==================== 후처리: 손가락 방향 분석 ====================
def analyze_finger_direction(results):
    """
    오른손 검지 방향 분석

    Returns:
    --------
    direction: str
        'up', 'down', 'unknown'
    finger_tip_y: float
        검지 끝 y 좌표
    """
    if results.right_hand_landmarks is None:
        return "unknown", None

    # 오른손 랜드마크
    hand = results.right_hand_landmarks.landmark

    # 검지 손가락
    index_tip = hand[8]  # 검지 끝
    index_mcp = hand[5]  # 검지 base (손바닥 쪽)

    # 방향 판단: 끝이 base보다 위에 있으면 '위'
    if index_tip.y < index_mcp.y - 0.05:  # 5% 이상 차이
        direction = "up"
    elif index_tip.y > index_mcp.y + 0.05:
        direction = "down"
    else:
        direction = "unknown"  # 수평

    return direction, index_tip.y


def check_hand_above_shoulder(results):
    """
    손이 어깨 위에 있는지 확인

    Returns:
    --------
    is_above: bool
        True if 손이 어깨 위
    """
    if results.right_hand_landmarks is None or results.pose_landmarks is None:
        return False

    # 오른손 손목
    right_wrist = results.right_hand_landmarks.landmark[0]  # 손목

    # 어깨
    pose = results.pose_landmarks.landmark
    right_shoulder = pose[SHOULDER_RIGHT_IDX]

    # 손목이 어깨보다 위에 있나?
    return right_wrist.y < right_shoulder.y


def analyze_hand_movement(sequence_buffer):
    """
    손의 전체 움직임 방향 분석

    Parameters:
    -----------
    sequence_buffer: deque
        최근 30 프레임의 랜드마크

    Returns:
    --------
    movement: str
        'up', 'down', 'unknown'
    """
    if len(sequence_buffer) < 10:
        return "unknown"

    # 최근 프레임들에서 오른손 y 좌표 추출
    # 오른손 검지 끝: index 63+8*3+1 = 88 (y 좌표)
    recent_frames = list(sequence_buffer)[-10:]  # 최근 10 프레임

    y_positions = []
    for frame in recent_frames:
        # 오른손 검지 끝 y 좌표
        right_index_y = frame[63 + 8 * 3 + 1]  # 오른손 시작(63) + 검지(8) * 3 + y(1)
        y_positions.append(right_index_y)

    # 전체 움직임 판단
    start_y = np.mean(y_positions[:3])  # 처음 3개
    end_y = np.mean(y_positions[-3:])  # 마지막 3개

    if end_y < start_y - 0.1:  # 위로 이동 (y는 작을수록 위)
        return "up"
    elif end_y > start_y + 0.1:  # 아래로 이동
        return "down"
    else:
        return "unknown"


# ==================== 후처리: soundup/sounddown 보정 ====================
def postprocess_sound_prediction(
    class_name, confidence, results, sequence_buffer, pred_probs, class_idx
):
    """
    sounddown 예측에 대한 후처리

    Parameters:
    -----------
    class_name: str
        예측된 클래스
    confidence: float
        예측 신뢰도
    results: MediaPipe results
        현재 프레임의 랜드마크
    sequence_buffer: deque
        최근 프레임들
    pred_probs: np.array
        전체 클래스 확률
    class_idx: int
        예측된 클래스 인덱스

    Returns:
    --------
    corrected_class: str
    corrected_confidence: float
    corrected_idx: int
    correction_reason: str
    """
    # sounddown이 아니면 그대로 반환
    if class_name != "sounddown":
        return class_name, confidence, class_idx, ""

    # soundup 인덱스
    soundup_idx = CLASS_MAPPING.get("soundup")
    if soundup_idx is None:
        return class_name, confidence, class_idx, ""

    # soundup 신뢰도
    soundup_conf = float(pred_probs[0][soundup_idx])

    # ===== Rule 1: 검지 위 + 어깨 위 → soundup =====
    finger_dir, finger_y = analyze_finger_direction(results)
    is_above_shoulder = check_hand_above_shoulder(results)
    hand_movement = analyze_hand_movement(sequence_buffer)

    print("   [후처리 분석]")
    print(f"   - 검지 방향: {finger_dir}")
    print(f"   - 어깨 위: {is_above_shoulder}")
    print(f"   - 전체 움직임: {hand_movement}")
    print(f"   - soundup 신뢰도: {soundup_conf*100:.1f}%")

    # 조건 1: 검지가 위를 가리키고 + 어깨 위 + 위로 움직임
    if finger_dir == "up" and is_above_shoulder and hand_movement == "up":
        # soundup 신뢰도가 일정 수준 이상이면
        if soundup_conf > 0.3:  # 30% 이상이면 가능성 있음
            print("   보정: sounddown → soundup")
            print("      이유: 검지 위 + 어깨 위 + 위로 움직임")
            return "soundup", soundup_conf, soundup_idx, "finger_up"

    # 조건 2: 검지가 아래 + 아래로 움직임 → sounddown 확정
    if finger_dir == "down" and hand_movement == "down":
        print("   확정: sounddown 유지")
        print("      이유: 검지 아래 + 아래로 움직임")
        return class_name, confidence, class_idx, "confirmed_down"

    # 그 외: 원본 그대로
    return class_name, confidence, class_idx, ""


# ==================== 시각화 ====================
def draw_landmarks_on_frame(frame, results):
    """프레임에 랜드마크 그리기"""
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2),
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=2),
        )

        # 검지 강조
        hand = results.right_hand_landmarks.landmark
        index_tip = hand[8]
        h, w = frame.shape[:2]
        x, y = int(index_tip.x * w), int(index_tip.y * h)
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # 초록색 큰 점

    if results.face_landmarks:
        lm = results.face_landmarks.landmark
        h, w = frame.shape[:2]
        for idx in FACE_INDICES:
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

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

        cv2.line(frame, points[0], points[1], (0, 255, 255), 2)
        cv2.line(frame, points[0], points[2], (0, 255, 255), 2)
        cv2.line(frame, points[2], points[4], (0, 255, 255), 2)
        cv2.line(frame, points[1], points[3], (0, 255, 255), 2)
        cv2.line(frame, points[3], points[5], (0, 255, 255), 2)

    return frame


# ==================== 메인 추론 ====================
def main(video_path, show_video=True):
    """비디오에서 수화 인식 + 후처리"""

    print("=" * 80)
    print("비디오 기반 수화 인식 + soundup/sounddown 후처리")
    print("=" * 80)
    print(f"비디오: {video_path}\n")

    # MediaPipe 초기화
    print("[1/4] MediaPipe Holistic 초기화...")
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=MP_MODEL_COMPLEXITY,
        refine_face_landmarks=MP_REFINE_FACE_LANDMARKS,
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_TRACKING_CONFIDENCE,
    )
    print("초기화 완료")

    # 모델 로드
    print("\n[2/4] 모델 로드...")
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일 없음: {MODEL_PATH}")
        holistic.close()
        return

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

    # 비디오 로드
    print("\n[3/4] 비디오 로드...")
    if not os.path.exists(video_path):
        print(f"비디오 파일 없음: {video_path}")
        holistic.close()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 열기 실패")
        holistic.close()
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("비디오 로드 완료")
    print(f"   FPS: {fps}, 총 프레임: {total_frames}")

    # 시퀀스 버퍼
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # 통계
    frame_count = 0
    prediction_history = []
    correction_history = []
    last_prediction = None
    last_confidence = 0

    # 최고 신뢰도 추적
    max_confidence = 0
    max_confidence_class = None
    max_confidence_idx = None

    # 추론
    print("\n[4/4] 추론 시작...")
    print(f"   시퀀스 길이: {SEQUENCE_LENGTH}")
    print(f"   신뢰도 임계값: {CONFIDENCE_THRESHOLD}")
    print("   후처리: soundup/sounddown 활성화")
    print("=" * 80)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            landmarks = extract_holistic_landmarks(results)

            if landmarks is not None:
                sequence_buffer.append(landmarks)

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    sequence = np.array(list(sequence_buffer))
                    sequence_with_velocity = add_velocity_to_sequence(sequence)

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

                    # 후처리 적용!
                    corrected_name, corrected_conf, corrected_idx, reason = (
                        postprocess_sound_prediction(
                            class_name, confidence, results, sequence_buffer, pred, class_idx
                        )
                    )

                    # 보정 여부 출력
                    if reason:
                        print(
                            f"{corrected_idx}_{corrected_name}_({corrected_conf*100:.1f}%) "
                            f"[보정: {reason}]"
                        )
                        correction_history.append(
                            {"original": class_name, "corrected": corrected_name, "reason": reason}
                        )
                    else:
                        print(f"{corrected_idx}_{corrected_name}_({corrected_conf*100:.1f}%)")

                    # 결과 저장 (보정된 결과로)
                    prediction_history.append((corrected_name, corrected_conf))

                    if corrected_name.lower() != "none" and corrected_conf > max_confidence:
                        max_confidence = corrected_conf
                        max_confidence_class = corrected_name
                        max_confidence_idx = corrected_idx

                    last_prediction = corrected_name
                    last_confidence = corrected_conf

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
        print(f"후처리 보정: {len(correction_history)}회")

        # 보정 상세
        if correction_history:
            print("\n후처리 보정 상세:")
            for corr in correction_history:
                print(f"   {corr['original']} → {corr['corrected']} ({corr['reason']})")

        # none 제외 필터링
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
            result_count = most_common[1]["count"]
            result_avg_conf = np.mean(most_common[1]["confidences"])
            result_idx = CLASS_MAPPING.get(result)

            # 클래스별 출현 통계
            print("\n" + "-" * 80)
            print("클래스별 출현 통계 (none 제외)")
            print("-" * 80)

            sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]["count"], reverse=True)

            for class_name, stats in sorted_classes:
                class_idx = CLASS_MAPPING.get(class_name)
                avg_conf = np.mean(stats["confidences"])
                max_conf = max(stats["confidences"])
                min_conf = min(stats["confidences"])

                print(
                    f"  {class_idx}_{class_name:12s}: {stats['count']:3d}회 "
                    f"(평균: {avg_conf*100:5.1f}%, 최고: {max_conf*100:5.1f}%, "
                    f"최저: {min_conf*100:5.1f}%)"
                )

            # 최종 결과
            print("\n" + "=" * 80)
            print("최종 결과")
            print("=" * 80)

            print("\n[1] 가장 많이 출현한 클래스:")
            print("-" * 80)
            print(f"   예측 클래스: {result}")
            print(f"   클래스 인덱스: {result_idx}")
            print(
                f"   출현 횟수: {result_count}회 ({result_count/len(filtered_predictions)*100:.1f}%)"
            )
            print(f"   평균 신뢰도: {result_avg_conf*100:.2f}%")

            if max_confidence_class:
                print("\n[2] 가장 높은 신뢰도:")
                print("-" * 80)
                print(f"   예측 클래스: {max_confidence_class}")
                print(f"   클래스 인덱스: {max_confidence_idx}")
                print(f"   최고 신뢰도: {max_confidence*100:.2f}%")

            print("\n[3] 최종 판단:")
            print("=" * 80)

            if result == max_confidence_class:
                print("일치! 예측 결과가 일관성 있습니다.")
                print(f"   결론: {result_idx}_{result}")

                # 스마일 이모티콘 조건 체크
                if result_avg_conf >= 0.9:
                    print(f"\n평균 신뢰도 {result_avg_conf*100:.1f}% - 매우 정확합니다!")
            else:
                print("불일치!")
                print(
                    f"   A) 빈도 기준: {result_idx}_{result} "
                    f"({result_count}회, 평균 {result_avg_conf*100:.1f}%)"
                )
                print(
                    f"   B) 최고 신뢰도: {max_confidence_idx}_{max_confidence_class} "
                    f"(최고 {max_confidence*100:.1f}%)"
                )
                print(f"\n 권장: 빈도가 더 높은 '{result}'을(를) 최종 결과로 선택")

                # 스마일 이모티콘 조건 체크
                if result_avg_conf >= 0.9:
                    print(f"\n평균 신뢰도 {result_avg_conf*100:.1f}% - 매우 정확합니다!")

            print("=" * 80)
            return result
        else:
            print("\n최종 결과: 'none' 이외의 클래스가 감지되지 않았습니다.")
            print("=" * 80)
            return "none"


if __name__ == "__main__":
    # ========================================
    # 비디오 경로 설정
    # ========================================
    VIDEO_PATH = "./data/raw_videos/test/soundup_5_2.mp4"  # ← 여기를 수정하세요!
    # ========================================

    SHOW_VIDEO = True

    print(f"\n설정된 비디오: {VIDEO_PATH}\n")

    result = main(video_path=VIDEO_PATH, show_video=SHOW_VIDEO)

    if result:
        print(f"\n추론 완료! 최종 결과: {result}")
