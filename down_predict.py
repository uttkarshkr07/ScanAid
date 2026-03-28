# Complete script for loading the model and making a prediction
# Includes face detection, cropping, and alignment

import tensorflow as tf
import numpy as np
import cv2  # OpenCV
import sys

# --- 1. Load your saved model ---
try:
    model = tf.keras.models.load_model('down_syndrome_detector.h5')
except IOError:
    print("Error: Model file 'down_syndrome_detector.h5' not found.")
    print("Please run train.py first to create the model.")
    sys.exit()

# --- 2. Load and Pre-process the New Image ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
image_path = 'test_image.jpg' # Image to test

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image, detects the face, crops it with a margin, 
    resizes to model dimensions, and normalizes pixels.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

img_normalized = preprocess_image(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))

if img_normalized is None:
    print(f"Error: Test image '{image_path}' not found or could not be loaded.")
    print("Please add a test_image.jpg to the folder.")
    sys.exit()

img_batch = np.expand_dims(img_normalized, axis=0)

# --- 3. Make the Prediction ---
prediction = model.predict(img_batch)
prediction_value = prediction[0][0]

print("--- Prediction ---")
print(f"Image: {image_path}")
print(f"Raw model output (sigmoid): {prediction_value}")

# --- 4. Interpret the Result ---
# This assumes 'down_syndrome_features' is class 0 (lower value)
# and 'typical_features' is class 1 (higher value).
if prediction_value < 0.5:
    confidence = (1 - prediction_value) * 100
    print(f"Result: Detected 'Down Syndrome Features' (Confidence: {confidence:.2f}%)")
else:
    confidence = prediction_value * 100
    print(f"Result: Detected 'Typical Features' (Confidence: {confidence:.2f}%)")
