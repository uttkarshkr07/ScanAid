"""
ScanAid: Model Evaluation Module

Computes classification metrics and saves a confusion matrix for a trained model.
Supports both the standard Down Syndrome classifier and the Siamese Angelman model.

Usage — standard classifier:
    python src/evaluate.py \\
        --model models/down_syndrome_detector.h5 \\
        --test-data data/processed/test/ \\
        --model-type standard

Usage — Siamese model (requires a reference set to compare against):
    python src/evaluate.py \\
        --model models/angelman_siamese_model.h5 \\
        --test-data data/processed/test/ \\
        --model-type siamese \\
        --reference-data syndrome_dataset/ \\
        --distance-threshold 0.5
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def _load_image(path, target_size=(224, 224)):
    """Load and normalize a single image to float32 [0, 1]."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img.astype('float32') / 255.0


def _collect_images(directory):
    """
    Walk directory and return (paths, labels, class_names).
    Subfolders are treated as class names; label = sorted index.
    """
    class_names = sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ])
    if not class_names:
        raise ValueError(f"No class subfolders found in: {directory}")

    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(directory, cls)
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in _IMAGE_EXTS:
                paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)

    return paths, labels, class_names


def _plot_confusion_matrix(cm, class_names, save_path):
    """Save a confusion matrix as a PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Predicted label',
        ylabel='True label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to: {save_path}")


def _print_metrics(y_true, y_pred, y_score, class_names):
    """Print all metrics and return a dict."""
    acc = accuracy_score(y_true, y_pred)

    # For binary classification: positive class = 0 (syndrome), negative = 1 (typical)
    if len(class_names) == 2:
        sensitivity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        specificity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        try:
            auc = roc_auc_score(y_true, 1.0 - y_score)  # lower score = syndrome class
        except ValueError:
            auc = float('nan')
    else:
        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
        specificity = float('nan')
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        auc = float('nan')

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Sensitivity  : {sensitivity:.4f}  (recall for syndrome class)")
    print(f"  Specificity  : {specificity:.4f}  (recall for typical class)")
    print(f"  Precision    : {precision:.4f}")
    print(f"  F1 Score     : {f1:.4f}")
    print(f"  AUC-ROC      : {auc:.4f}")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return dict(accuracy=acc, sensitivity=sensitivity, specificity=specificity,
                precision=precision, f1=f1, auc_roc=auc)


# ---------------------------------------------------------------------------
# Standard classifier evaluation
# ---------------------------------------------------------------------------
def evaluate_standard(model_path, test_data_dir, output_dir):
    """
    Evaluate a standard binary/multiclass Keras classifier.
    Loads images from test_data_dir (one subfolder per class), runs inference,
    and computes all metrics.
    """
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    paths, y_true, class_names = _collect_images(test_data_dir)
    print(f"Evaluating on {len(paths)} images across classes: {class_names}")

    y_score = []
    failed = 0
    valid_indices = []

    for i, path in enumerate(paths):
        img = _load_image(path)
        if img is None:
            print(f"  [Skip] Cannot load: {path}")
            failed += 1
            continue
        batch = np.expand_dims(img, axis=0)
        pred = model.predict(batch, verbose=0)[0]
        y_score.append(float(pred[0]) if pred.shape[-1] == 1 else float(pred.max()))
        valid_indices.append(i)

    if failed:
        print(f"[Warning] {failed} images could not be loaded and were skipped.")

    y_true = [y_true[i] for i in valid_indices]
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = _print_metrics(y_true, y_pred, y_score, class_names)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(output_dir, "confusion_matrix_down_syndrome.png")
    _plot_confusion_matrix(cm, class_names, cm_path)

    return metrics


# ---------------------------------------------------------------------------
# Siamese model evaluation
# ---------------------------------------------------------------------------
def evaluate_siamese(model_path, test_data_dir, reference_data_dir,
                     distance_threshold=0.5, output_dir="reports"):
    """
    Evaluate the Siamese Angelman model using a 1-NN distance approach.

    For each test image:
      1. Compute distances to all reference images from each class.
      2. Predict the class with the lowest average distance.

    Args:
        model_path:          Path to the saved Siamese .h5 model.
        test_data_dir:       Directory with class subfolders of test images.
        reference_data_dir:  Directory with class subfolders of reference images
                             (can be the training set itself).
        distance_threshold:  Not used for 1-NN; kept for future threshold-based eval.
        output_dir:          Where to save the confusion matrix plot.
    """
    print(f"Loading Siamese model: {model_path}")

    # EuclideanDistance is a custom layer — must be declared before loading
    class EuclideanDistance(tf.keras.layers.Layer):
        def call(self, inputs):
            x, y = inputs
            sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
            return tf.math.sqrt(tf.math.maximum(sum_square, 1e-8))

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'contrastive_loss': _dummy_loss,
            'EuclideanDistance': EuclideanDistance,
        }
    )

    # Build reference embeddings per class using the base_network (one branch)
    base_network = _extract_base_network(model)
    if base_network is None:
        raise RuntimeError(
            "Could not extract base_network from the Siamese model. "
            "Ensure the model was saved with the full architecture."
        )

    ref_paths, _, ref_class_names = _collect_images(reference_data_dir)
    _, _, test_class_names = _collect_images(test_data_dir)

    # Use union of class names, sorted
    class_names = sorted(set(ref_class_names) | set(test_class_names))
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print(f"Building reference embeddings from: {reference_data_dir}")
    ref_embeddings = {cls: [] for cls in class_names}
    for path in ref_paths:
        cls = os.path.basename(os.path.dirname(path))
        img = _load_image(path)
        if img is None:
            continue
        emb = base_network.predict(np.expand_dims(img, 0), verbose=0)[0]
        ref_embeddings[cls].append(emb)

    # Average reference embedding per class (prototype)
    class_prototypes = {}
    for cls, embs in ref_embeddings.items():
        if embs:
            class_prototypes[cls] = np.mean(embs, axis=0)
        else:
            print(f"[Warning] No reference images for class '{cls}'.")

    test_paths, y_true_raw, _ = _collect_images(test_data_dir)
    test_class_names_list = [
        os.path.basename(os.path.dirname(p)) for p in test_paths
    ]
    y_true = [class_to_idx[c] for c in test_class_names_list]

    print(f"Evaluating {len(test_paths)} test images...")
    y_pred, y_score, valid_indices = [], [], []

    for i, path in enumerate(test_paths):
        img = _load_image(path)
        if img is None:
            continue
        emb = base_network.predict(np.expand_dims(img, 0), verbose=0)[0]

        # Distance to each class prototype
        distances = {}
        for cls, proto in class_prototypes.items():
            dist = float(np.linalg.norm(emb - proto))
            distances[cls] = dist

        predicted_cls = min(distances, key=distances.get)
        y_pred.append(class_to_idx[predicted_cls])

        # Score = distance to the "syndrome" class (lower = more likely syndrome)
        syndrome_cls = [c for c in class_names if c != 'typical'][0]
        y_score.append(distances.get(syndrome_cls, 0.0))
        valid_indices.append(i)

    y_true = [y_true[i] for i in valid_indices]
    y_score = np.array(y_score)

    metrics = _print_metrics(y_true, y_pred, y_score, class_names)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(output_dir, "confusion_matrix_angelman.png")
    _plot_confusion_matrix(cm, class_names, cm_path)

    return metrics


def _dummy_loss(y_true, y_pred):
    """Placeholder to satisfy custom_objects during model loading."""
    return tf.reduce_mean(y_pred)


def _extract_base_network(siamese_model):
    """Extract the shared feature extractor sub-model from a Siamese model."""
    for layer in siamese_model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name == 'feature_extractor':
            return layer
    # Fallback: first sub-model found
    for layer in siamese_model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser():
    parser = argparse.ArgumentParser(
        description="ScanAid model evaluation: accuracy, sensitivity, specificity, "
                    "precision, F1, AUC-ROC, and confusion matrix."
    )
    parser.add_argument('--model', required=True, metavar='PATH',
                        help='Path to the trained .h5 model file.')
    parser.add_argument('--test-data', required=True, metavar='DIR',
                        help='Directory with class subfolders of test images.')
    parser.add_argument('--model-type', choices=['standard', 'siamese'],
                        default='standard',
                        help='Type of model: standard classifier or Siamese network.')
    parser.add_argument('--reference-data', metavar='DIR', default=None,
                        help='(Siamese only) Directory with reference class images.')
    parser.add_argument('--distance-threshold', type=float, default=0.5,
                        help='(Siamese only) Distance threshold for classification.')
    parser.add_argument('--output-dir', metavar='DIR', default='reports',
                        help='Directory to save confusion matrix plot. Default: reports/')
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_type == 'standard':
        evaluate_standard(args.model, args.test_data, args.output_dir)
    else:
        if not args.reference_data:
            print("Error: --reference-data is required for Siamese model evaluation.")
            exit(1)
        evaluate_siamese(
            model_path=args.model,
            test_data_dir=args.test_data,
            reference_data_dir=args.reference_data,
            distance_threshold=args.distance_threshold,
            output_dir=args.output_dir,
        )
