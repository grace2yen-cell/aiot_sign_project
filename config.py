"""
config.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
수화 기반 IoT 제어 프로젝트 - 전체 설정
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

프로젝트 정보:
- 모델: MediaPipe Holistic + LSTM
- 특징: 상대 좌표 + Velocity 피처
- 클래스: 9개 (수화 8 + none 1)
- 차원: 336 (원본 168 + 속도 168) ⭐ 수정됨!

작성일: 2025.11.16
버전: 2.1
"""

import os

# ==================== 프로젝트 기본 정보 ====================
PROJECT_NAME = "Sign Language IoT Control"
PROJECT_VERSION = "2.1"
CURRENT_PHASE = 1  # 1: 5명 (4 Train + 1 Val) | 2: 6명 (5 Train + 1 Val)

# ==================== 경로 설정 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 경로
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_VIDEO_DIR = os.path.join(DATA_DIR, 'raw_videos')
TRAIN_DIR = os.path.join(RAW_VIDEO_DIR, 'train')
VAL_DIR = os.path.join(RAW_VIDEO_DIR, 'val')

# 처리된 데이터 경로
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_data')
AUGMENTED_DIR = os.path.join(DATA_DIR, 'augmented_data')

# 모델 및 로그 경로
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
VIZ_DIR = os.path.join(BASE_DIR, 'visualization')

# 파일 경로
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.h5')

# ==================== 클래스 설정 ====================
CLASS_NAMES = [
    'lighton',      # 0: 조명 켜기 (양손 가슴 → 위로 펴기)
    'lightoff',     # 1: 조명 끄기 (양손 가슴 → 아래로 쥐기)
    'tempup',       # 2: 온도 올리기 (한손 어깨 아래 → 목/얼굴까지)
    'tempdown',     # 3: 온도 내리기 (한손 목/얼굴 → 어깨 아래)
    'soundup',      # 4: 소리 올리기 (한손 귀 옆 → 위로)
    'sounddown',    # 5: 소리 내리기 (한손 귀 옆 → 아래로)
    'blindup',      # 6: 블라인드 올리기 (양손 눈 앞 → 위로)
    'blinddown',    # 7: 블라인드 내리기 (양손 눈 앞 → 아래로)
    'none'          # 8: 수화가 아닌 동작 (40개 일상 동작)
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_MAPPING = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ==================== MediaPipe 설정 ====================
# Holistic 설정
MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE = 0.5
MP_MODEL_COMPLEXITY = 1
MP_REFINE_FACE_LANDMARKS = False

# Face 랜드마크 인덱스 (8개 핵심 포인트)
FACE_INDICES = [
    1,      # 코 끝 (중심 기준점)
    33,     # 왼쪽 눈 (blind 구분용)
    263,    # 오른쪽 눈 (blind 구분용)
    10,     # 윗입술 중앙
    152,    # 턱 끝 (temp/blind 구분용)
    234,    # 왼쪽 관자놀이 (얼굴 폭)
    454,    # 오른쪽 관자놀이 (얼굴 폭)
    199     # 턱 중앙
]

# Pose 기준점 (상대 좌표 변환용)
SHOULDER_LEFT_IDX = 11      # 왼쪽 어깨
SHOULDER_RIGHT_IDX = 12     # 오른쪽 어깨
HIP_LEFT_IDX = 23          # 왼쪽 골반
HIP_RIGHT_IDX = 24         # 오른쪽 골반

# ==================== 데이터 설정 ====================
# 시퀀스 설정
SEQUENCE_LENGTH = 30        # 프레임 수 (1초)
TARGET_FPS = 30            # 목표 FPS

# ⭐⭐⭐ 피처 차원 (수정됨!)
HAND_DIM = 21 * 3 * 2      # 126 (양손: 왼손 63 + 오른손 63)
FACE_DIM = 8 * 3           # 24 (선택된 8개 포인트)
POSE_DIM = 6 * 3           # 18 (어깨 2 + 팔꿈치 2 + 손목 2)

FEATURE_DIM_RAW = HAND_DIM + FACE_DIM + POSE_DIM  # 168 (원본)
FEATURE_DIM_VELOCITY = FEATURE_DIM_RAW             # 168 (속도)
FEATURE_DIM_FINAL = FEATURE_DIM_RAW + FEATURE_DIM_VELOCITY  # 336 ⭐ 수정!

# None 클래스 설정
NONE_ACTIONS = 40          # None 동작 개수 (1개 영상에 40개 포함)
NONE_DURATION = 3          # 각 동작 길이 (초)

# ==================== 정규화 설정 ====================
NORMALIZE_TO_RELATIVE = True    # 상대 좌표 사용 ⭐
ADD_VELOCITY = True            # 속도 피처 추가 ⭐
USE_Z_SCORE_NORM = True        # Z-score 정규화 사용

# ==================== 증강 설정 ====================
# 증강 목표 개수
AUGMENT_TARGET = {
    'sign_train': 600,      # Train: 각 수화 클래스
    'none_train': 4800,     # Train: none
    'sign_val': 150,        # ⭐ NEW! Val: 각 수화 클래스
    'none_val': 1200        # ⭐ NEW! Val: none
}

# 증강 기법 활성화
USE_SPEED_CHANGE = True
USE_TEMPORAL_NOISE = True
USE_SPATIAL_SCALE = True
USE_ROTATION = True
USE_VELOCITY_PERTURB = True
USE_PAUSE = True

# 증강 파라미터
SPEED_CHANGE_RANGE = (0.8, 1.2)          # 느리게/빠르게
TEMPORAL_NOISE_LEVEL = 0.015             # 시간축 노이즈
SPATIAL_SCALE_RANGE = (0.85, 1.15)       # 공간 스케일
ROTATION_ANGLE_RANGE = (-10, 10)         # 회전 각도 (도)
VELOCITY_PERTURB_RATIO = 0.15            # 속도 섭동 비율
PAUSE_PROBABILITY = 0.1                  # 멈춤 확률
PAUSE_STRENGTH = 0.2                     # 멈춤 강도

# ==================== 학습 설정 ====================
# LSTM 구조
LSTM_UNITS_1 = 256         # ⭐ 128 → 256 (더 복잡한 패턴 학습 능력)
LSTM_UNITS_2 = 128         # ⭐ 64 → 128
DENSE_UNITS = 64           # ⭐ 32 → 64
DROPOUT_RATE = 0.3         # ⭐ 0.6 → 0.3 (너무 높음! 학습 방해)

USE_BIDIRECTIONAL = True
L2_REGULARIZATION = 0.0005  # ⭐ 0.001 → 0.0005 (약간 완화)

# 학습 파라미터
EPOCHS = 150               # ⭐ 100 → 150
BATCH_SIZE = 16            # ⭐ 32 → 16 (안정적 학습)
LEARNING_RATE = 0.0005     # ⭐ 0.001 → 0.0005 (천천히)

# 콜백 설정
EARLY_STOPPING_PATIENCE = 30     # ⭐ 15 → 30 (더 기다리기)
REDUCE_LR_PATIENCE = 10          # ⭐ 5 → 10
REDUCE_LR_FACTOR = 0.3           # ⭐ 0.5 → 0.3 (더 천천히 감소)
MIN_LEARNING_RATE = 1e-7         # ⭐ 1e-6 → 1e-7

# ==================== 추론 설정 ====================
# Threshold 설정
CONFIDENCE_THRESHOLD = 0.85         # 최소 신뢰도
PREDICTION_SMOOTHING = 3           # 예측 평활화 (프레임 수)

# 후보정 설정
POST_PROCESS_THRESHOLDS = {
    'temp_start_max': 0.50,        # temp는 y < 0.50에서 시작
    'blind_start_min': 0.55,       # blind는 y > 0.55에서 시작
    'confidence_min': 0.85         # 신뢰도 낮을 때만 보정
}

# ==================== 서버 설정 (선택사항) ====================
TCP_HOST = '0.0.0.0'
TCP_PORT = 9999
UDP_PORT = 8888

# 서버 저장용 파일명 (클라이언트로부터 받은 비디오 임시 저장)
SAVE_NAME = "received.avi"

# ==================== 출력 설정 ====================
# 디렉토리 자동 생성
for directory in [MODEL_DIR, LOG_DIR, VIZ_DIR, PROCESSED_DIR, AUGMENTED_DIR]:
    os.makedirs(directory, exist_ok=True)

# 클래스별 폴더 생성
for class_name in CLASS_NAMES:
    class_dir = os.path.join(PROCESSED_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

# ==================== 정보 출력 ====================
if __name__ == '__main__':
    print("=" * 60)
    print(f"📌 {PROJECT_NAME} v{PROJECT_VERSION}")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")

    print(f"\n📊 Classes: {NUM_CLASSES}")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"   {idx}: {name}")

    print(f"\n🎯 Features:")
    print(f"   Hands: {HAND_DIM}")
    print(f"   Face: {FACE_DIM} (8개 포인트)")
    print(f"   Pose: {POSE_DIM}")
    print(f"   Raw Total: {FEATURE_DIM_RAW}")
    print(f"   + Velocity: {FEATURE_DIM_VELOCITY}")
    print(f"   = Final: {FEATURE_DIM_FINAL} ⭐")

    print(f"\n🎯 Model Settings:")
    print(f"   Sequence Length: {SEQUENCE_LENGTH}")
    print(f"   Input Shape: ({SEQUENCE_LENGTH}, {FEATURE_DIM_FINAL})")
    print(f"   LSTM Units: {LSTM_UNITS_1} → {LSTM_UNITS_2}")
    print(f"   Dense Units: {DENSE_UNITS}")
    print(f"   Dropout: {DROPOUT_RATE}")
    print(f"   Bidirectional: {USE_BIDIRECTIONAL}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")

    print(f"\n🎯 Augmentation:")
    print(f"   Sign classes: {AUGMENT_TARGET['sign']} samples each")
    print(f"   None class: {AUGMENT_TARGET['none']} samples")
    print(f"   Total Train: {AUGMENT_TARGET['sign'] * 8 + AUGMENT_TARGET['none']} samples")
    print(f"   Ratio: {AUGMENT_TARGET['sign'] * 8}:{AUGMENT_TARGET['none']} = 1:1")

    # GPU 확인
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"\n💻 GPU Available: {len(gpus)} device(s)")
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("   ⚠️  No GPU detected - will use CPU")
    except Exception as e:
        print(f"\n⚠️  TensorFlow not loaded yet")

    print("\n" + "=" * 60)
    print("✅ Config loaded successfully!")
    print("=" * 60)