"""
ScanAid: Grad-CAM Explainability Module

Generates a Grad-CAM heatmap showing which facial regions contributed most
to the model's prediction, then overlays it on the original image.

Works with:
  - Standard classifier (e.g. Down Syndrome model)
  - Siamese model (applies Grad-CAM to the shared feature extractor branch)

Targets the last Conv2D layer of the MobileNetV2 backbone automatically.

Usage — standard model:
    python src/gradcam.py \\
        --model models/down_syndrome_detector.h5 \\
        --image test.jpg \\
        --output reports/gradcam_output.jpg

Usage — Siamese model:
    python src/gradcam.py \\
        --model models/angelman_siamese_model.h5 \\
        --image test.jpg \\
        --output reports/gradcam_angelman.jpg \\
        --model-type siamese
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf


# ---------------------------------------------------------------------------
# Core Grad-CAM logic
# ---------------------------------------------------------------------------

def find_last_conv_layer(model):
    """
    Traverse model layers in reverse and return the name of the last Conv2D.
    For nested models (e.g. MobileNetV2 inside a wrapper), recurse into sub-models.
    """
    for layer in reversed(model.layers):
        # Recurse into sub-models (e.g. MobileNetV2 embedded in base_network)
        if isinstance(layer, tf.keras.Model):
            name = find_last_conv_layer(layer)
            if name:
                return name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Compute a Grad-CAM heatmap for a single image.

    Args:
        model:                 Keras model (standard classifier or base_network).
        img_array:             Float32 array of shape (1, H, W, 3), normalized [0,1].
        last_conv_layer_name:  Name of the target Conv2D layer.
        pred_index:            Class index to explain. If None, uses the argmax prediction.

    Returns:
        heatmap: float32 array of shape (H, W), values in [0, 1].
    """
    # Build a sub-model that outputs both the conv activations and the final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            # For sigmoid output (binary): use raw output
            # For softmax output (multi-class): use argmax class
            if predictions.shape[-1] == 1:
                pred_index = 0
                score = predictions[:, 0]
            else:
                pred_index = tf.argmax(predictions[0])
                score = predictions[:, pred_index]
        else:
            score = predictions[:, pred_index]

    # Gradient of the score w.r.t. conv layer output
    grads = tape.gradient(score, conv_outputs)

    # Pool gradients across spatial dimensions → importance weight per filter
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv output channels by their importance
    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img_path, heatmap, output_path, alpha=0.4):
    """
    Overlay a Grad-CAM heatmap on the original image and save to output_path.

    Args:
        original_img_path: Path to the original (un-preprocessed) image.
        heatmap:           float32 array of shape (H, W), values in [0, 1].
        output_path:       Where to save the overlaid image.
        alpha:             Opacity of the heatmap overlay (0 = invisible, 1 = opaque).
    """
    img = cv2.imread(original_img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {original_img_path}")

    # Resize heatmap to match the original image
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend with original image
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    print(f"Grad-CAM overlay saved to: {output_path}")


def load_and_preprocess(image_path, target_size=(224, 224)):
    """Load an image and normalize to [0, 1] float32 for model input."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img.astype('float32') / 255.0


# ---------------------------------------------------------------------------
# Model-type specific entry points
# ---------------------------------------------------------------------------

def run_gradcam_standard(model_path, image_path, output_path):
    """
    Apply Grad-CAM to a standard binary/multiclass MobileNetV2-based classifier.
    """
    model = tf.keras.models.load_model(model_path)

    last_conv = find_last_conv_layer(model)
    if last_conv is None:
        raise RuntimeError("No Conv2D layer found in the model.")
    print(f"Targeting conv layer: {last_conv}")

    img = load_and_preprocess(image_path)
    img_batch = np.expand_dims(img, axis=0)

    prediction = model.predict(img_batch, verbose=0)[0]
    pred_value = float(prediction[0]) if prediction.shape[-1] == 1 else float(prediction.max())
    print(f"Model prediction score: {pred_value:.4f}")

    heatmap = make_gradcam_heatmap(model, img_batch, last_conv)
    overlay_heatmap(image_path, heatmap, output_path)


def run_gradcam_siamese(model_path, image_path, output_path):
    """
    Apply Grad-CAM to the Siamese model's shared feature extractor.

    The Siamese model outputs a distance (not a class probability), so Grad-CAM
    is computed on the base_network (feature extractor) branch using the L2 norm
    of the embedding as the score. This highlights which facial regions the network
    uses to build its feature representation.
    """
    def _dummy_loss(y_true, y_pred):
        return tf.reduce_mean(y_pred)

    siamese_model = tf.keras.models.load_model(
        model_path, custom_objects={'contrastive_loss': _dummy_loss}
    )

    # Extract base_network (feature extractor)
    base_network = None
    for layer in siamese_model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name == 'feature_extractor':
            base_network = layer
            break
    if base_network is None:
        for layer in siamese_model.layers:
            if isinstance(layer, tf.keras.Model):
                base_network = layer
                break
    if base_network is None:
        raise RuntimeError("Could not extract base_network from Siamese model.")

    last_conv = find_last_conv_layer(base_network)
    if last_conv is None:
        raise RuntimeError("No Conv2D layer found in base_network.")
    print(f"Targeting conv layer: {last_conv}")

    img = load_and_preprocess(image_path)
    img_batch = np.expand_dims(img, axis=0)

    # Use embedding L2 norm as the scalar score for Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=base_network.inputs,
        outputs=[
            base_network.get_layer(last_conv).output,
            base_network.output,
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, embedding = grad_model(img_batch)
        score = tf.norm(embedding)  # scalar: "how strong is this embedding"

    grads = tape.gradient(score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    overlay_heatmap(image_path, heatmap.numpy(), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description="ScanAid Grad-CAM: visualize which facial regions drive the prediction."
    )
    parser.add_argument('--model', required=True, metavar='PATH',
                        help='Path to the trained .h5 model.')
    parser.add_argument('--image', required=True, metavar='PATH',
                        help='Path to the input face image.')
    parser.add_argument('--output', required=True, metavar='PATH',
                        help='Where to save the Grad-CAM overlay image.')
    parser.add_argument('--model-type', choices=['standard', 'siamese'],
                        default='standard',
                        help='Type of model. Default: standard.')
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()

    if args.model_type == 'standard':
        run_gradcam_standard(args.model, args.image, args.output)
    else:
        run_gradcam_siamese(args.model, args.image, args.output)
