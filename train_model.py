"""
train_model.py - VitaDetect Model Trainer
Run this script to retrain the model from your dataset.

Usage:
    python train_model.py --dataset path/to/dataset
    python train_model.py               # uses ./dataset/ by default

Dataset structure expected:
    dataset/
        Vitamin A/  (images)
        Vitamin B/  (images)
        ...
"""
import os
import sys
import pickle
import argparse
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

IMG_SIZE = (64, 64)
OUTPUT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vitamodel.pkl")

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
    h2d, _, _ = np.histogram2d(arr[:,:,0].ravel(), arr[:,:,1].ravel(), bins=16, range=[[0,255],[0,255]])
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
    return np.array(features, dtype=np.float32)

def augment(img):
    return [
        img,
        ImageOps.mirror(img),
        ImageEnhance.Brightness(img).enhance(1.2),
        ImageEnhance.Contrast(img).enhance(1.2),
    ]

def load_dataset(dataset_dir):
    X, y = [], []
    classes = sorted([c for c in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, c))])
    if not classes:
        print(f"ERROR: No class folders found in {dataset_dir}")
        sys.exit(1)
    print(f"Classes found: {classes}")
    for label in classes:
        folder = os.path.join(dataset_dir, label)
        imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        print(f"  {label}: {len(imgs)} images")
        for fname in imgs:
            try:
                img = Image.open(os.path.join(folder, fname)).convert("RGB")
                for aug_img in augment(img):
                    X.append(extract_features(aug_img))
                    y.append(label)
            except Exception as e:
                print(f"    Skipping {fname}: {e}")
    return np.array(X), np.array(y), classes

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"))
args = parser.parse_args()

if not os.path.exists(args.dataset):
    print(f"ERROR: Dataset folder not found: {args.dataset}")
    sys.exit(1)

print(f"\nTraining from: {args.dataset}")
print("Extracting features (with augmentation)...")
X, y, classes = load_dataset(args.dataset)
print(f"Total samples (augmented): {len(X)}, Feature size: {X.shape[1]}")

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\nTraining SVM classifier...")
clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
acc = (y_pred == y_test).mean()
print(f"\nTest Accuracy: {acc*100:.1f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))

with open(OUTPUT_MODEL, 'wb') as f:
    pickle.dump({'scaler': scaler, 'clf': clf, 'label_encoder': le, 'classes': classes}, f)

print(f"Model saved → {OUTPUT_MODEL}")
print("\nDone! Restart your Flask app to use the new model.")
