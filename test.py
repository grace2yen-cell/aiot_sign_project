import numpy as np
from tensorflow.keras.models import load_model

print("=" * 60)
print("🔍 현재 상태 확인")
print("=" * 60)

# 1. 데이터 확인
try:
    X_train = np.load('./data/augmented_data/X_train.npy')
    print(f"\n✅ 학습 데이터: {X_train.shape}")
    data_dim = X_train.shape[2]
except:
    print("\n❌ 학습 데이터 없음")
    data_dim = None

# 2. 모델 확인
try:
    model = load_model('models/lstm_model.h5')
    print(f"✅ 모델 로드 성공")
    print(f"   Input shape: {model.input_shape}")
    model_dim = model.input_shape[2]
except Exception as e:
    print(f"\n❌ 모델 로드 실패: {e}")
    model_dim = None

# 3. 판정
print("\n" + "=" * 60)
if data_dim and model_dim:
    if data_dim == model_dim:
        print(f"✅ 정상! 데이터({data_dim})와 모델({model_dim}) 일치")
        print("\n👉 재학습 불필요! config.py만 교체하고 바로 사용 가능!")
    else:
        print(f"❌ 불일치! 데이터({data_dim}) vs 모델({model_dim})")
        print("\n👉 재학습 필요! 5_train_lstm.py 다시 실행하세요!")
else:
    print("⚠️  확인 불가")
print("=" * 60)
# ```
#
# ---
#
# ## 📊 예상되는 결과와 대응
#
# ### **Case A: 정상 케이스** ✅
# ```
# ✅ 학습 데이터: (7200, 30, 336)
# ✅ 모델 로드 성공
#    Input shape: (None, 30, 336)
#
# ✅ 정상! 데이터(336)와 모델(336) 일치
# 👉 재학습 불필요!
# ```
#
# **→ 이 경우:**
# - config.py만 교체
# - 바로 8_video_test_basic.py 실행 가능!
# - **재학습 안 해도 됨!** 😊
#
# ---
#
# ### **Case B: 차원 불일치** ❌
# ```
# ✅ 학습 데이터: (7200, 30, 336)
# ✅ 모델 로드 성공
#    Input shape: (None, 30, 498)
#
# ❌ 불일치! 데이터(336) vs 모델(498)
# 👉 재학습 필요!
# ```
#
# **→ 이 경우:**
# - config.py 교체 (FEATURE_DIM_FINAL = 336)
# - `python 5_train_lstm.py` 재실행
# - **재학습 필요!** 😢
#
# ---
#
# ### **Case C: 모델 로드 에러** 💥
# ```
# ✅ 학습 데이터: (7200, 30, 336)
# ❌ 모델 로드 실패: ...