"""
5_train_lstm.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 5: LSTM 모델 학습 (개선 버전)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 목표:
- Bidirectional LSTM + Multi-Head Attention
- 입력: (30, 336) = 30 프레임 × 336 차원
- 출력: 9개 클래스 (수화 8 + none 1)

⭐ 개선사항:
1. Dropout 감소 (0.6 → 0.3) - 더 정교한 학습
2. 모델 크기 증가 (LSTM 128/64 → 256/128)
3. Multi-Head Attention 추가 - 중요 프레임 집중
4. Class Weight 적용 - 혼동 클래스 강화
5. Adam Optimizer - 안정적인 학습

📊 입력:
- augmented_data/X_train.npy: (9600, 30, 336)
- augmented_data/y_train.npy: (9600,)
- augmented_data/X_val.npy: (2400, 30, 336)
- augmented_data/y_val.npy: (2400,)

📊 출력:
- models/lstm_model.h5: 학습된 모델
- logs/training_history.png: 학습 곡선
- logs/training_results.txt: 학습 결과
"""

import gpu_config  # GPU 설정

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional,
    MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
import time
from datetime import timedelta
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


# ==================== 정규화 ====================
def normalize_data(X_train, X_val):
    """Z-score 정규화"""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    # 정규화 파라미터 저장
    np.savez(os.path.join(AUGMENTED_DIR, 'normalization_params.npz'),
             mean=mean, std=std)

    return X_train_norm, X_val_norm


# ==================== 클래스 가중치 계산 ====================
def compute_custom_class_weights(y_train):
    """
    클래스별 가중치 계산 + 혼동 클래스 가중치 증가

    Parameters:
    -----------
    y_train: np.array
        학습 레이블 (9600,)

    Returns:
    --------
    class_weight_dict: dict
        클래스별 가중치
    """
    # 기본 가중치 (불균형 보정)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # ⭐ 혼동되는 클래스 가중치 증가
    problematic_classes = [
        'soundup', 'sounddown',  # sound vs blind vs temp
        'blindup',                # blind vs sound
        'lighton', 'lightoff'     # light on vs off
    ]

    print("\n클래스별 가중치:")
    print("=" * 60)

    for class_idx in range(NUM_CLASSES):
        class_name = CLASS_NAMES[class_idx]

        # 혼동 클래스는 1.5배 증가
        if class_name in problematic_classes:
            class_weight_dict[class_idx] *= 1.5
            marker = "⭐"
        else:
            marker = "  "

        weight = class_weight_dict[class_idx]
        print(f"{marker} {class_name:12s}: {weight:.3f}")

    print("=" * 60)

    return class_weight_dict


# ==================== 모델 구축 (Attention 추가) ====================
def build_lstm_model_with_attention(input_shape, num_classes):
    """
    Bidirectional LSTM + Multi-Head Attention

    Parameters:
    -----------
    input_shape: tuple
        (30, 336)
    num_classes: int
        9

    Returns:
    --------
    model: tf.keras.Model
    """
    inputs = Input(shape=input_shape)

    # ===== Layer 1: Bidirectional LSTM =====
    x = Bidirectional(
        LSTM(LSTM_UNITS_1,
             return_sequences=True,  # ⭐ Attention을 위해 시퀀스 유지
             kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))
    )(inputs)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    # ===== ⭐ Multi-Head Attention Layer =====
    # 중요한 프레임에 집중!
    attention_output = MultiHeadAttention(
        num_heads=8,                    # 8개의 attention head
        key_dim=LSTM_UNITS_1 // 4,      # key dimension
        dropout=0.1
    )(x, x)  # self-attention

    # Residual connection + Layer Normalization
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)

    # ===== Layer 2: Bidirectional LSTM =====
    x = Bidirectional(
        LSTM(LSTM_UNITS_2,
             return_sequences=False,
             kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    # ===== Dense Layers =====
    x = Dense(DENSE_UNITS,
              activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))(x)
    x = Dropout(0.3)(x)

    # ===== Output =====
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='LSTM_Attention')

    return model


# ==================== 학습 곡선 시각화 ====================
def plot_history(history, save_dir):
    """학습 곡선 그리기"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()


# ==================== 학습 결과 저장 ====================
def save_results(save_dir, history, train_acc, val_acc, train_loss, val_loss,
                 train_duration, X_train_shape, X_val_shape):
    """학습 결과 txt 저장"""
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'training_results.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("LSTM + Attention 모델 학습 결과\n")
        f.write("=" * 60 + "\n\n")

        # 데이터 정보
        f.write("📊 데이터 정보\n")
        f.write(f"Train Shape: {X_train_shape}\n")
        f.write(f"Val Shape: {X_val_shape}\n")
        f.write(f"Feature Dim: {FEATURE_DIM_FINAL}\n")
        f.write(f"Sequence Length: {SEQUENCE_LENGTH}\n\n")

        # 모델 구조
        f.write("🏗️  모델 구조\n")
        f.write(f"Bidirectional LSTM ({LSTM_UNITS_1}) + Dropout ({DROPOUT_RATE})\n")
        f.write(f"Multi-Head Attention (8 heads) ⭐\n")
        f.write(f"Residual Connection + Layer Normalization ⭐\n")
        f.write(f"Bidirectional LSTM ({LSTM_UNITS_2}) + Dropout ({DROPOUT_RATE})\n")
        f.write(f"Dense ({DENSE_UNITS}) + Dropout (0.3)\n")
        f.write(f"Dense ({NUM_CLASSES}, softmax)\n\n")

        # 학습 파라미터
        f.write("⚙️  학습 파라미터\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {len(history.history['accuracy'])}\n")
        f.write(f"L2 Regularization: {L2_REGULARIZATION}\n")
        f.write(f"Optimizer: Adam ⭐\n")
        f.write(f"Class Weight: Applied ⭐\n\n")

        # 최종 성능
        f.write("📊 최종 성능\n")
        f.write(f"Train Accuracy: {train_acc * 100:.2f}%\n")
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Val Accuracy: {val_acc * 100:.2f}%\n")
        f.write(f"Val Loss: {val_loss:.4f}\n")
        f.write(f"Overfitting Gap: {(train_acc - val_acc) * 100:.2f}%p\n\n")

        # 최고 성능
        best_val_acc = np.max(history.history['val_accuracy'])
        best_val_epoch = np.argmax(history.history['val_accuracy']) + 1

        f.write("🏆 최고 성능\n")
        f.write(f"Best Val Accuracy: {best_val_acc * 100:.2f}% (Epoch {best_val_epoch})\n")
        f.write(f"Training Time: {str(timedelta(seconds=int(train_duration)))}\n")

        # 스마일 이모티콘 조건
        if val_acc >= 0.98:
            f.write(f"\n😊 Val Accuracy {val_acc * 100:.1f}% - 매우 우수합니다!\n")

        f.write("=" * 60 + "\n")


# ==================== 메인 ====================
def main():
    total_start = time.time()

    print("=" * 60)
    print("🚀 Step 5: LSTM + Attention 모델 학습")
    print("=" * 60)

    # [1] 데이터 로드
    print("\n[1/7] 데이터 로드 중...")

    X_train = np.load(os.path.join(AUGMENTED_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(AUGMENTED_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(AUGMENTED_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(AUGMENTED_DIR, 'y_val.npy'))

    print(f"✅ Train: {X_train.shape}")
    print(f"✅ Val: {X_val.shape}")

    # Train:Val 비율 확인
    total_samples = len(X_train) + len(X_val)
    train_ratio = len(X_train) / total_samples * 100
    val_ratio = len(X_val) / total_samples * 100
    print(f"📊 Train:Val = {train_ratio:.1f}:{val_ratio:.1f}")

    # [2] 정규화
    print("\n[2/7] 데이터 정규화 중...")

    if USE_Z_SCORE_NORM:
        X_train_norm, X_val_norm = normalize_data(X_train, X_val)
        print("✅ Z-score 정규화 완료")
    else:
        X_train_norm = X_train
        X_val_norm = X_val
        print("⚠️  정규화 건너뜀")

    # [3] One-hot encoding
    print("\n[3/7] One-hot encoding...")

    y_train_onehot = to_categorical(y_train, NUM_CLASSES)
    y_val_onehot = to_categorical(y_val, NUM_CLASSES)

    print("✅ Encoding 완료")

    # [4] 클래스 가중치 계산
    print("\n[4/7] 클래스 가중치 계산 중...")

    class_weight_dict = compute_custom_class_weights(y_train)

    # [5] 모델 구축
    print("\n[5/7] 모델 구축 중...")

    model = build_lstm_model_with_attention(
        input_shape=(SEQUENCE_LENGTH, FEATURE_DIM_FINAL),
        num_classes=NUM_CLASSES
    )

    # ⭐ Adam Optimizer 사용
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n모델 구조:")
    model.summary()

    # [6] 콜백 설정
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # [7] 학습
    print("\n[6/7] 모델 학습 중...")
    print(f"⚙️  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
    print(f"⚙️  Learning Rate: {LEARNING_RATE}")
    print(f"⚙️  Dropout: {DROPOUT_RATE}")
    print(f"⚙️  LSTM Units: {LSTM_UNITS_1} → {LSTM_UNITS_2}")
    print(f"⚙️  Attention: Multi-Head (8 heads)")
    print(f"⚙️  Optimizer: Adam")
    print(f"⚙️  Class Weight: Applied")

    train_start = time.time()

    history = model.fit(
        X_train_norm, y_train_onehot,
        validation_data=(X_val_norm, y_val_onehot),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # ⭐ 클래스 가중치 적용
        verbose=1
    )

    train_duration = time.time() - train_start

    # [8] 최종 평가
    print("\n[7/7] 최종 평가...")

    train_loss, train_acc = model.evaluate(X_train_norm, y_train_onehot, verbose=0)
    val_loss, val_acc = model.evaluate(X_val_norm, y_val_onehot, verbose=0)

    print(f"\n✅ Train Accuracy: {train_acc * 100:.2f}%")
    print(f"✅ Val Accuracy: {val_acc * 100:.2f}%")
    print(f"⏱️  학습 시간: {str(timedelta(seconds=int(train_duration)))}")

    # ⭐ 스마일 이모티콘 조건
    if val_acc >= 0.98:
        print(f"\n😊 Val Accuracy {val_acc * 100:.1f}% - 매우 우수합니다!")

    # 시각화 및 저장
    plot_history(history, LOG_DIR)

    save_results(LOG_DIR, history, train_acc, val_acc, train_loss, val_loss,
                 train_duration, X_train.shape, X_val.shape)

    total_duration = time.time() - total_start

    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print("=" * 60)
    print(f"💾 모델 저장: {MODEL_PATH}")
    print(f"📊 로그 저장: {LOG_DIR}")
    print(f"⏱️  총 소요 시간: {str(timedelta(seconds=int(total_duration)))}")

    # 개선사항 요약
    print("\n🎯 적용된 개선사항:")
    print(f"   1. ⭐ Dropout 감소: 0.6 → {DROPOUT_RATE}")
    print(f"   2. ⭐ LSTM 크기 증가: 128/64 → {LSTM_UNITS_1}/{LSTM_UNITS_2}")
    print(f"   3. ⭐ Multi-Head Attention (8 heads)")
    print(f"   4. ⭐ Class Weight (혼동 클래스 1.5배)")
    print(f"   5. ⭐ Adam Optimizer")
    print(f"   6. ⭐ Val 데이터 증강 (75 → {len(X_val)}개)")

    print("\n💡 다음 단계:")
    print("   - 6_evaluate_model_val_.py로 성능 평가")
    print("   - 8_video_test_basic.py로 추론 테스트")
    print("=" * 60)


if __name__ == '__main__':
    main()