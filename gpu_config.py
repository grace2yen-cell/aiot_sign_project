# gpu_config.py
import os

USER_GPU = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = USER_GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 경고 숨기기

print(f"사용할 GPU: {USER_GPU}")
print("="*60)