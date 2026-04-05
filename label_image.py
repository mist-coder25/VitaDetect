"""
label_image.py  -  VitaDetect inference module (sklearn-based)
Uses color histogram + texture + spatial features with SVM classifier.
No TensorFlow required.
"""
import os
import pickle
import numpy as np
from PIL import Image

_model_data = None
IMG_SIZE = (64, 64)


def _base_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _initialize():
    global _model_data
    if _model_data is not None:
        return
    model_file = os.path.join(_base_dir(), "vitamodel.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"\n[VitaDetect] Model file not found: {model_file}\n"
            "Place vitamodel.pkl in the project root folder."
        )
    print("[VitaDetect] Loading model...")
    with open(model_file, "rb") as f:
        _model_data = pickle.load(f)
    print(f"[VitaDetect] Model ready. Classes: {_model_data['classes']}")


def extract_features(img):
    img = img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    features = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=64, range=(0, 255))
        features.extend(hist / (hist.sum() + 1e-7))
    gray = np.mean(arr, axis=2)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    grad = np.sqrt(gx**2 + gy**2)
    hist, _ = np.histogram(grad.ravel(), bins=32, range=(0, 200))
    features.extend(hist / (hist.sum() + 1e-7))
    h2d, _, _ = np.histogram2d(
        arr[:, :, 0].ravel(), arr[:, :, 1].ravel(),
        bins=16, range=[[0, 255], [0, 255]]
    )
    features.extend((h2d / (h2d.sum() + 1e-7)).ravel())
    for ch in range(3):
        d = arr[:, :, ch].ravel() / 255.0
        features.extend([d.mean(), d.std(), np.percentile(d, 25), np.percentile(d, 75)])
    h, w = arr.shape[:2]
    bh, bw = h // 8, w // 8
    for i in range(8):
        for j in range(8):
            block = arr[i*bh:(i+1)*bh, j*bw:(j+1)*bw, :]
            features.extend(block.mean(axis=(0, 1)) / 255)
    return np.array(features, dtype=np.float32).reshape(1, -1)


def main(img_path):
    """Classify a skin/tissue image. Returns predicted label e.g. 'Vitamin A'."""
    _initialize()
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"[VitaDetect] Image not found: {img_path}")
    img = Image.open(img_path).convert("RGB")
    feat = extract_features(img)
    scaler = _model_data['scaler']
    clf = _model_data['clf']
    le = _model_data['label_encoder']
    feat_scaled = scaler.transform(feat)
    proba = clf.predict_proba(feat_scaled)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = le.classes_[pred_idx]
    confidence = float(proba[pred_idx])
    print("\n[VitaDetect] Prediction scores:")
    for i, cls in enumerate(le.classes_):
        bar = "=" * int(proba[i] * 40)
        print(f"  {cls:<14}: {proba[i]*100:5.1f}%  [{bar}]")
    print(f"[VitaDetect] Best: '{pred_label}' ({confidence*100:.1f}%)\n")
    if confidence < 0.25:
        raise ValueError(
            f"Low confidence ({confidence*100:.1f}%). "
            "Please upload a clearer skin/tissue image."
        )
    return pred_label
