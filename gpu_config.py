"""
gpu_config.py
[실제 사용] GPU / TensorFlow 환경 설정 (학습·추론 스크립트에서 import)
"""

import os

# 특정 GPU만 사용하려면 아래 두 줄의 주석을 풀고 번호를 지정하세요.
# (예: 여러 GPU가 있는 서버에서 3번 GPU만 사용)
# USER_GPU = "3"
# os.environ['CUDA_VISIBLE_DEVICES'] = USER_GPU

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow 경고 숨기기
