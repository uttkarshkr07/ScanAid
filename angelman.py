"""
ScanAid: Angelman Syndrome Static Module
Architecture: Siamese Neural Network using MobileNetV2 backbone.

This script defines a few-shot learning model designed to compare a patient's 
facial image against a known reference image to detect subtle phenotypic 
markers of Angelman Syndrome. It includes face detection, alignment, 
and a custom data generator for pair-wise training.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import cv2  # pip install opencv-python
import os
import random
import math
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. Base Network (Feature Extractor)
# ==========================================
def build_base_network(input_shape=(224, 224, 3)):
    """
    Creates the base feature extraction network.
    We use MobileNetV2 because it is lightweight and optimized for Edge devices.
    """
    base_model = applications.MobileNetV2(
        input_shape=input_shape, 
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze the base model weights initially
    base_model.trainable = False 

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation=None)(x)  # 128-dimensional embedding

    # L2 Normalization — use UnitNormalization layer (Keras 3 compatible)
    outputs = layers.UnitNormalization(axis=1)(x)

    return Model(inputs, outputs, name="feature_extractor")

# ==========================================
# 2. Distance Layer and Loss Function
# ==========================================
class EuclideanDistance(layers.Layer):
    """
    Keras 3-compatible layer that computes the euclidean distance between
    two embedding vectors. Replaces layers.Lambda which is deprecated in Keras 3.
    """
    def call(self, inputs):
        x, y = inputs
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, 1e-8))


def contrastive_loss(y_true, y_pred):
    """
    Contrastive Loss.
    y_true = 1 (similar faces), y_true = 0 (dissimilar faces).
    """
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    
    return tf.math.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# ==========================================
# 3. Siamese Network Assembly
# ==========================================
def build_siamese_model(base_network, input_shape=(224, 224, 3)):
    """
    Builds the twin-architecture Siamese model.

    Output is the raw euclidean distance (NOT sigmoid) so it is compatible
    with contrastive_loss. At inference time, threshold the distance:
        distance < 0.5  →  similar (same syndrome)
        distance >= 0.5 →  dissimilar (different class)
    """
    input_1 = layers.Input(shape=input_shape, name="image_1")
    input_2 = layers.Input(shape=input_shape, name="image_2")

    embedding_1 = base_network(input_1)
    embedding_2 = base_network(input_2)

    # Raw distance — contrastive_loss expects this directly, not a sigmoid output
    distance = EuclideanDistance(name="distance")([embedding_1, embedding_2])

    return Model(inputs=[input_1, input_2], outputs=distance, name="ScanAid_Angelman_Siamese")

# ==========================================
# 4. Image Preprocessing & Face Alignment
# ==========================================
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image, detects the face, crops it with a margin, 
    resizes to MobileNetV2 dimensions, and normalizes pixels.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use OpenCV's Haar Cascade for lightweight face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 0:
        # Get the largest face detected
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Add a 10% margin around the face so we don't crop too tightly
        margin_w, margin_h = int(w * 0.1), int(h * 0.1)
        x1, y1 = max(0, x - margin_w), max(0, y - margin_h)
        x2, y2 = min(img_rgb.shape[1], x + w + margin_w), min(img_rgb.shape[0], y + h + margin_h)
        
        face_crop = img_rgb[y1:y2, x1:x2]
    else:
        # Fallback if the detector misses the face: use the center of the image
        h, w = img_rgb.shape[:2]
        crop_size = min(h, w)
        start_y, start_x = (h - crop_size) // 2, (w - crop_size) // 2
        face_crop = img_rgb[start_y:start_y+crop_size, start_x:start_x+crop_size]

    # Resize and normalize [0, 1]
    face_resized = cv2.resize(face_crop, target_size)
    face_normalized = face_resized.astype('float32') / 255.0
    
    return face_normalized

# ==========================================
# 5. Augmentation Helper (for minority class)
# ==========================================
def augment_image(img):
    """
    Apply random augmentations to a normalized float32 image array.
    Used to synthetically expand the small Angelman class during training.
    Augmentations: horizontal flip, brightness jitter, rotation, random crop-and-resize.
    """
    h, w = img.shape[:2]

    # Random horizontal flip
    if random.random() < 0.5:
        img = np.fliplr(img)

    # Random brightness jitter (±20%)
    delta = random.uniform(-0.2, 0.2)
    img = np.clip(img + delta, 0.0, 1.0)

    # Random rotation ±15 degrees
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # Random crop-and-resize (crop to 80–100% of image, then resize back)
    crop_frac = random.uniform(0.80, 1.0)
    crop_h, crop_w = int(h * crop_frac), int(w * crop_frac)
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    img = img[top:top + crop_h, left:left + crop_w]
    img = cv2.resize(img, (w, h))

    return img.astype('float32')


# ==========================================
# 6. Custom Siamese Data Generator
# ==========================================
def siamese_data_generator(data_dir, batch_size=16, target_size=(224, 224),
                            minority_class='angelman', max_typical_ratio=3,
                            augment_minority=True, class_paths_override=None):
    """
    Yields batches of image pairs and their similarity labels.
    Assumed folder structure:
    data_dir/
      ├── angelman/ (images of angelman syndrome)
      └── typical/  (images of typical development)

    Args:
        class_paths_override: dict mapping class name → list of image paths.
            When provided, data_dir is ignored for path loading (used for
            train/val splits without touching the filesystem).

    Imbalance handling:
      - typical pool is capped at min(actual_count, angelman_count * max_typical_ratio)
        so the model is not dominated by typical-typical pairs.
      - Angelman images are augmented on-the-fly when augment_minority=True.
      - Same-class pairs explicitly alternate between classes so each class is
        represented equally in the same-class half of each batch.
    """
    if class_paths_override is not None:
        class_paths = {cls: list(paths) for cls, paths in class_paths_override.items()}
        classes = list(class_paths.keys())
    else:
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        class_paths = {
            cls: [os.path.join(data_dir, cls, f) for f in os.listdir(os.path.join(data_dir, cls))
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for cls in classes
        }

    # --- Cap the majority class so it doesn't dwarf the minority class ---
    if minority_class in class_paths:
        minority_count = len(class_paths[minority_class])
        cap = minority_count * max_typical_ratio
        for cls in classes:
            if cls != minority_class and len(class_paths[cls]) > cap:
                class_paths[cls] = random.sample(class_paths[cls], cap)
                print(f"[Imbalance fix] Capped '{cls}' pool to {cap} images "
                      f"({max_typical_ratio}x the {minority_count} '{minority_class}' images).")

    while True:
        batch_img_1, batch_img_2, batch_labels = [], [], []
        # Alternate same-class pair class to ensure equal representation
        same_class_turn = 0

        while len(batch_labels) < batch_size:
            is_same_class = random.choice([True, False])

            if is_same_class:
                # Strict alternation: pick angelman turn then typical turn
                cls = classes[same_class_turn % len(classes)]
                same_class_turn += 1
                if len(class_paths[cls]) < 2:
                    continue
                img1_path, img2_path = random.sample(class_paths[cls], 2)
                label = 1.0
            else:
                if len(classes) < 2:
                    continue
                cls1, cls2 = random.sample(classes, 2)
                img1_path = random.choice(class_paths[cls1])
                img2_path = random.choice(class_paths[cls2])
                label = 0.0

            img1 = preprocess_image(img1_path, target_size)
            img2 = preprocess_image(img2_path, target_size)

            if img1 is None or img2 is None:
                continue

            # Augment minority class images to increase effective variety
            if augment_minority:
                if minority_class in img1_path:
                    img1 = augment_image(img1)
                if minority_class in img2_path:
                    img2 = augment_image(img2)

            batch_img_1.append(img1)
            batch_img_2.append(img2)
            batch_labels.append(label)

        yield (np.array(batch_img_1), np.array(batch_img_2)), np.array(batch_labels)

# ==========================================
# 7. Main Execution Block
# ==========================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    IMAGE_SHAPE = (224, 224, 3)
    BATCH_SIZE = 16
    EPOCHS = 30           # Phase 1: frozen backbone
    FINE_TUNE_EPOCHS = 10 # Phase 2: unfreeze last 30 MobileNetV2 layers
    DATA_DIRECTORY = "syndrome_dataset"
    MODEL_SAVE_PATH = os.path.join("models", "angelman_siamese_model.h5")

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    if not os.path.exists(DATA_DIRECTORY):
        print(f"\n[Error] Dataset folder '{DATA_DIRECTORY}' not found.")
        print("Expected subfolders: angelman/ and typical/")
        exit(1)

    angelman_dir = os.path.join(DATA_DIRECTORY, "angelman")
    angelman_count = len([
        f for f in os.listdir(angelman_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    # steps_per_epoch: enough to cover all unique angelman-angelman pairs once per epoch
    # C(n,2) unique pairs / batch_size, minimum 10
    steps_per_epoch = max(10, math.comb(angelman_count, 2) // BATCH_SIZE)
    val_steps = max(5, steps_per_epoch // 5)
    print(f"Angelman images: {angelman_count} | steps_per_epoch: {steps_per_epoch} | val_steps: {val_steps}")

    # --- Train / validation split (80/20) per class ---
    # Explicitly restrict to angelman vs typical — ignore down_syndrome subfolder
    ANGELMAN_CLASSES = ['angelman', 'typical']
    train_class_paths = {}
    val_class_paths = {}
    for cls in ANGELMAN_CLASSES:
        cls_dir = os.path.join(DATA_DIRECTORY, cls)
        if not os.path.isdir(cls_dir):
            print(f"[Warning] Expected class folder not found: {cls_dir}")
            continue
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(imgs)
        split = int(len(imgs) * 0.8)
        train_class_paths[cls] = imgs[:split]
        val_class_paths[cls] = imgs[split:]

    # --- Class weights ---
    # Pair labels (0=dissimilar, 1=similar) are generated 50/50 by the generator,
    # so sklearn gives ~1:1 weights. We compute it explicitly for transparency.
    pair_label_sample = np.array([0] * angelman_count + [1] * angelman_count)
    pair_weights = compute_class_weight('balanced', classes=np.array([0, 1]),
                                        y=pair_label_sample)
    class_weight_dict = {int(c): float(w) for c, w in enumerate(pair_weights)}
    print(f"Siamese pair class weights: {class_weight_dict}")

    # --- Callbacks ---
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ]

    print("\nBuilding Siamese Architecture...")
    base_network = build_base_network(input_shape=IMAGE_SHAPE)
    siamese_model = build_siamese_model(base_network, input_shape=IMAGE_SHAPE)

    siamese_model.compile(
        loss=contrastive_loss,
        optimizer=Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    siamese_model.summary()

    print(f"\nDataset found at '{DATA_DIRECTORY}'. Initializing generators...")
    train_generator = siamese_data_generator(
        DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        minority_class='angelman',
        max_typical_ratio=3,
        augment_minority=True,
        class_paths_override=train_class_paths
    )
    val_generator = siamese_data_generator(
        DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        minority_class='angelman',
        max_typical_ratio=3,
        augment_minority=False,
        class_paths_override=val_class_paths
    )

    # =============================================
    # PHASE 1: Train with frozen MobileNetV2 backbone
    # =============================================
    print("\n--- Phase 1: Training with frozen backbone ---")
    history1 = siamese_model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # =============================================
    # PHASE 2: Fine-tune — unfreeze last 30 MobileNetV2 layers
    # =============================================
    print("\n--- Phase 2: Fine-tuning last 30 MobileNetV2 layers ---")
    mobilenet_backbone = next(
        (l for l in base_network.layers if isinstance(l, tf.keras.Model)), None
    )
    if mobilenet_backbone is None:
        print("[Warning] Could not locate MobileNetV2 inside base_network. Skipping Phase 2.")
        history2 = None
    else:
        base_network.trainable = True
        mobilenet_backbone.trainable = True
        for layer in mobilenet_backbone.layers[:-30]:
            layer.trainable = False

        # 10x lower LR to avoid destroying pretrained features
        siamese_model.compile(
            loss=contrastive_loss,
            optimizer=Adam(learning_rate=1e-5),
            metrics=['accuracy']
        )

        history2 = siamese_model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=val_steps,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=callbacks
        )

    # --- Merge histories and plot ---
    def _concat(h1, h2, key):
        return h1.history.get(key, []) + (h2.history.get(key, []) if h2 else [])

    phase_boundary = len(history1.history['loss'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes, ['loss', 'accuracy'], ['Loss', 'Accuracy']):
        ax.plot(_concat(history1, history2, metric), label=f'Train {title}')
        ax.plot(_concat(history1, history2, f'val_{metric}'), label=f'Val {title}')
        ax.axvline(x=phase_boundary, color='gray', linestyle='--', label='Fine-tune start')
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plot_path = os.path.join("reports", "angelman_siamese_training.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved to '{plot_path}'")

    siamese_model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to '{MODEL_SAVE_PATH}'")