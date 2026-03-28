"""
ScanAid: Shared face detection and preprocessing utilities.

Run once to preprocess a raw dataset directory into data/processed/:
    python src/preprocessing.py --input ./syndrome_dataset --output ./data/processed/down_syndrome

The output images are saved as regular (non-normalized) JPEGs. Pixel normalization
to [0, 1] happens at training time inside each model's data generator.
"""

import os
import cv2
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Supported image extensions
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# Lazily load the Haar Cascade once per process
_face_cascade = None

def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from: {cascade_path}\n"
                "Make sure opencv-python is installed correctly."
            )
    return _face_cascade


def detect_and_crop_face(image_path: str, target_size: tuple = (224, 224)):
    """
    Load an image, detect the largest face, crop it with a 10% margin,
    and resize to target_size.

    Returns the cropped face as a uint8 RGB numpy array, or None if:
      - the image file could not be read
      - no face was detected (failure is logged, NOT silently replaced)

    Normalization to [0, 1] is intentionally omitted — callers handle that.

    Args:
        image_path: Absolute or relative path to the source image.
        target_size: (width, height) for the output image. Default: (224, 224).

    Returns:
        np.ndarray of shape (height, width, 3) dtype uint8, or None on failure.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Could not read image: %s", image_path)
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    face_cascade = _get_cascade()
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        # For medical data we must know exactly which images failed detection
        # so clinicians can review them manually — no silent fallback.
        logger.warning("No face detected: %s", image_path)
        return None

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    # 10% margin so we don't crop too tightly around the face
    margin_w = int(w * 0.1)
    margin_h = int(h * 0.1)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_rgb.shape[1], x + w + margin_w)
    y2 = min(img_rgb.shape[0], y + h + margin_h)

    face_crop = img_rgb[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, target_size)

    return face_resized  # uint8 RGB, not normalized


def preprocess_dataset(input_dir: str, output_dir: str, target_size: tuple = (224, 224)):
    """
    Run face detection and cropping on every image in input_dir ONCE and save
    the results to output_dir, preserving the subfolder structure.

    Example input layout:
        input_dir/
            down_syndrome/
                img001.jpg
            typical/
                img002.jpg

    Produces:
        output_dir/
            down_syndrome/
                img001.jpg   <- cropped face, saved as JPEG
            typical/
                img002.jpg

    Images where face detection fails are logged and skipped (not saved).
    A summary is printed at the end so you know the failure rate.

    Args:
        input_dir:   Root directory containing class subfolders.
        output_dir:  Destination root. Created if it doesn't exist.
        target_size: (width, height) for saved images. Default: (224, 224).
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    # Collect all image files (walk the full tree to support nested structure)
    all_images = [
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]

    if not all_images:
        logger.warning("No images found in: %s", input_path)
        return

    logger.info("Found %d images in '%s'", len(all_images), input_path)

    succeeded = 0
    failed = 0
    failed_paths = []

    for src_path in all_images:
        # Mirror the subfolder path under output_dir
        relative = src_path.relative_to(input_path)
        dst_path = output_path / relative
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        face = detect_and_crop_face(str(src_path), target_size=target_size)

        if face is None:
            failed += 1
            failed_paths.append(str(src_path))
            continue

        # Save as BGR (OpenCV convention) — convert back from RGB
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst_path), face_bgr)
        succeeded += 1

    # --- Summary ---
    total = len(all_images)
    logger.info("=" * 60)
    logger.info("Preprocessing complete.")
    logger.info("  Saved:  %d / %d", succeeded, total)
    logger.info("  Failed: %d / %d (face not detected)", failed, total)

    if failed_paths:
        logger.info("Images with no face detected:")
        for p in failed_paths:
            logger.info("  FAILED: %s", p)
        logger.info(
            "Review the failed images manually. Do not include them in training "
            "without verification — mislabeled or corrupted medical images can "
            "silently bias your model."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "ScanAid preprocessing: detect and crop faces from a raw dataset "
            "directory, saving results to an output directory."
        )
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="DIR",
        help="Root directory of the raw dataset (e.g. ./syndrome_dataset).",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="DIR",
        help="Output directory for cropped images (e.g. ./data/processed/down_syndrome).",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("W", "H"),
        help="Target image size as WIDTH HEIGHT. Default: 224 224.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    target = tuple(args.size)  # (width, height)
    preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=target,
    )
