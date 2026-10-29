"""
test.py
[실험용] 학습 데이터와 저장된 모델의 입력 차원(336)이 일치하는지 확인할 때 사용
(피처 차원을 498 -> 336으로 바꾼 뒤 재학습이 필요한지 점검하던 스크립트)
"""

import numpy as np
from tensorflow.keras.models import load_model

print("=" * 60)
print("현재 상태 확인")
print("=" * 60)

# 1. 데이터 확인
try:
    X_train = np.load("./data/augmented_data/X_train.npy")
    print(f"\n학습 데이터: {X_train.shape}")
    data_dim = X_train.shape[2]
except Exception:
    print("\n학습 데이터 없음")
    data_dim = None

# 2. 모델 확인
try:
    model = load_model("models/lstm_model.h5")
    print("모델 로드 성공")
    print(f"   Input shape: {model.input_shape}")
    model_dim = model.input_shape[2]
except Exception as e:
    print(f"\n모델 로드 실패: {e}")
    model_dim = None

# 3. 판정
print("\n" + "=" * 60)
if data_dim and model_dim:
    if data_dim == model_dim:
        print(f"정상! 데이터({data_dim})와 모델({model_dim}) 일치")
        print("\n재학습 불필요! config.py만 교체하고 바로 사용 가능!")
    else:
        print(f"불일치! 데이터({data_dim}) vs 모델({model_dim})")
        print("\n재학습 필요! 5_train_lstm.py 다시 실행하세요!")
else:
    print("확인 불가")
print("=" * 60)
