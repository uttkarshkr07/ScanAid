"""
ScanAid: Down Syndrome Static Classifier — Training Script
Architecture: MobileNetV2 with two-phase fine-tuning (frozen → partial unfreeze).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')  # save plots without a display
import matplotlib.pyplot as plt
import cv2

# --- 1. SETUP: Define Constants ---
DATA_DIR = 'syndrome_dataset'
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10            # Phase 1: frozen backbone
FINE_TUNE_EPOCHS = 10  # Phase 2: partial unfreeze

# --- 2. FACE CROP PREPROCESSING ---
def face_crop_preprocessing(img_array):
    """
    Custom preprocessing for ImageDataGenerator: detect and crop the largest face.
    Falls back to the full image if no face is detected.
    Returns float32 (rescale=1./255 in the generator handles normalization).
    """
    img_uint8 = img_array.astype('uint8')
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        margin_w, margin_h = int(w * 0.1), int(h * 0.1)
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_uint8.shape[1], x + w + margin_w)
        y2 = min(img_uint8.shape[0], y + h + margin_h)
        face_crop = img_uint8[y1:y2, x1:x2]
    else:
        face_crop = img_uint8

    return cv2.resize(face_crop, (IMG_WIDTH, IMG_HEIGHT)).astype('float32')


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # --- Data generators ---
    datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=face_crop_preprocessing,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True,
        classes=['down_syndrome', 'typical']  # explicit: ignore angelman subfolder
    )

    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        classes=['down_syndrome', 'typical']  # explicit: ignore angelman subfolder
    )

    print(f"Classes: {train_generator.class_indices}")
    print(f"Training samples: {train_generator.samples} | Validation samples: {validation_generator.samples}")

    # --- Class weights (handles imbalance) ---
    labels = train_generator.classes
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(weights.tolist()))
    print(f"Class weights: {class_weight_dict}")

    # --- Shared callbacks ---
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ]

    # =============================================
    # PHASE 1: Train with frozen MobileNetV2 backbone
    # =============================================
    base_model = MobileNetV2(
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    print("\n--- Phase 1: Training with frozen backbone ---")
    history1 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # =============================================
    # PHASE 2: Fine-tune — unfreeze last 30 layers
    # =============================================
    print("\n--- Phase 2: Fine-tuning last 30 MobileNetV2 layers ---")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with 10x lower learning rate to avoid destroying pretrained features
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # --- Merge histories for a single plot ---
    merged = {k: history1.history[k] + history2.history[k] for k in history1.history}
    phase_boundary = len(history1.history['accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, val_metric, title) in zip(axes, [
        ('accuracy', 'val_accuracy', 'Accuracy'),
        ('loss', 'val_loss', 'Loss'),
    ]):
        ax.plot(merged[metric], label=f'Train {title}')
        ax.plot(merged[val_metric], label=f'Val {title}')
        ax.axvline(x=phase_boundary, color='gray', linestyle='--', label='Fine-tune start')
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plot_path = os.path.join("reports", "down_syndrome_training.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved to '{plot_path}'")

    # --- Save model ---
    model_path = os.path.join("models", "down_syndrome_detector.h5")
    model.save(model_path)
    print(f"Model saved to '{model_path}'")
