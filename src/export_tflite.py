"""
ScanAid: TensorFlow Lite Export Module

Converts a trained Keras .h5 model to a TFLite file using float16 quantization.
Float16 quantization halves model size while preserving near-identical accuracy,
making it suitable for mobile/edge deployment (Android, Raspberry Pi, etc.).

For the Siamese Angelman model: exports only the base_network (feature extractor),
since TFLite does not support the Lambda layer used for euclidean distance.
At inference on-device, compute the distance between two embeddings manually.

Usage — standard model:
    python src/export_tflite.py \\
        --model models/down_syndrome_detector.h5 \\
        --output models/down_syndrome.tflite

Usage — Siamese model (exports base_network only):
    python src/export_tflite.py \\
        --model models/angelman_siamese_model.h5 \\
        --output models/angelman_embedding.tflite \\
        --model-type siamese
"""

import os
import argparse
import tensorflow as tf


def _file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def export_standard(model_path, output_path):
    """
    Convert a standard Keras classifier to TFLite with float16 quantization.

    Args:
        model_path:  Path to the .h5 model.
        output_path: Destination .tflite file path.
    """
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    print("Converting to TFLite (float16 quantization)...")
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    orig_size = _file_size_mb(model_path)
    tflite_size = _file_size_mb(output_path)
    reduction = (1 - tflite_size / orig_size) * 100

    print(f"\nExport complete.")
    print(f"  Original model : {orig_size:.2f} MB  ({model_path})")
    print(f"  TFLite model   : {tflite_size:.2f} MB  ({output_path})")
    print(f"  Size reduction : {reduction:.1f}%")


def export_siamese_base_network(model_path, output_path):
    """
    Extract the base_network (feature extractor) from a Siamese model and
    export it to TFLite with float16 quantization.

    The Lambda layer used for euclidean distance is not TFLite-compatible,
    so only the embedding branch is exported. At inference:
      1. Run the TFLite model on image A → embedding_A
      2. Run the TFLite model on image B → embedding_B
      3. distance = ||embedding_A - embedding_B||₂
      4. If distance < 0.5 → same syndrome; else → different class

    Args:
        model_path:  Path to the Siamese .h5 model.
        output_path: Destination .tflite file path.
    """
    def _dummy_loss(y_true, y_pred):
        return tf.reduce_mean(y_pred)

    print(f"Loading Siamese model: {model_path}")
    siamese_model = tf.keras.models.load_model(
        model_path,
        custom_objects={'contrastive_loss': _dummy_loss}
    )

    # Extract the feature extractor sub-model
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
        raise RuntimeError(
            "Could not find base_network inside the Siamese model. "
            "Ensure the model was saved with the full architecture."
        )

    print(f"Extracted sub-model: {base_network.name}  "
          f"(input: {base_network.input_shape}, output: {base_network.output_shape})")

    # Save base_network as a temporary .h5 for size comparison
    tmp_path = output_path.replace('.tflite', '_base_network_tmp.h5')
    base_network.save(tmp_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(base_network)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    print("Converting base_network to TFLite (float16 quantization)...")
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    orig_size = _file_size_mb(model_path)
    base_size = _file_size_mb(tmp_path)
    tflite_size = _file_size_mb(output_path)
    os.remove(tmp_path)

    print(f"\nExport complete.")
    print(f"  Full Siamese model   : {orig_size:.2f} MB  ({model_path})")
    print(f"  Base network (.h5)   : {base_size:.2f} MB")
    print(f"  TFLite (base network): {tflite_size:.2f} MB  ({output_path})")
    print(f"  Size reduction       : {(1 - tflite_size / orig_size) * 100:.1f}%  vs full model")
    print()
    print("On-device inference instructions:")
    print("  1. Load this TFLite model and run inference on image A → embedding_A")
    print("  2. Run inference on image B → embedding_B")
    print("  3. distance = sqrt(sum((embedding_A - embedding_B) ** 2))")
    print("  4. distance < 0.5 → same syndrome class; distance >= 0.5 → different class")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description="ScanAid TFLite export: convert a trained .h5 model to "
                    "TFLite with float16 quantization for edge deployment."
    )
    parser.add_argument('--model', required=True, metavar='PATH',
                        help='Path to the trained .h5 model.')
    parser.add_argument('--output', required=True, metavar='PATH',
                        help='Output path for the .tflite file.')
    parser.add_argument('--model-type', choices=['standard', 'siamese'],
                        default='standard',
                        help='Type of model. Siamese exports only the base_network. '
                             'Default: standard.')
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        exit(1)

    if args.model_type == 'standard':
        export_standard(args.model, args.output)
    else:
        export_siamese_base_network(args.model, args.output)
