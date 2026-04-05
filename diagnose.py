"""
diagnose.py  -  Run this to test your model directly from terminal.

Usage:
    python diagnose.py path/to/your/test_image.jpg

This shows you EXACTLY what the model predicts and with what confidence,
so you can tell if the problem is the model, the image, or the app.
"""
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python diagnose.py <path_to_image>")
    print("Example: python diagnose.py test.jpg")
    sys.exit(1)

img = sys.argv[1]

if not os.path.exists(img):
    print(f"ERROR: Image file not found: {img}")
    sys.exit(1)

print(f"\nTesting image: {img}")
print("="*50)

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ── Load model ──────────────────────────────────────────────────────────────
model_file = "retrained_graph.pb"
label_file = "retrained_labels.txt"

if not os.path.exists(model_file):
    print(f"ERROR: {model_file} not found. Place it in this folder.")
    sys.exit(1)

print("Loading model...")
graph = tf.Graph()
graph_def = tf.compat.v1.GraphDef()
with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
with graph.as_default():
    tf.import_graph_def(graph_def)

labels = []
for line in tf.gfile.GFile(label_file).readlines():
    s = line.rstrip()
    if s:
        labels.append(s)

print(f"Labels found: {labels}")
print()

# ── Preprocess image ─────────────────────────────────────────────────────────
img_path = os.path.abspath(img).replace("\\", "/")

prep_graph = tf.Graph()
with prep_graph.as_default():
    file_reader  = tf.io.read_file(img_path, "file_reader")
    if img_path.lower().endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3)
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3)
    float_caster  = tf.cast(image_reader, tf.float32)
    dims_expander  = tf.expand_dims(float_caster, 0)
    resized        = tf.image.resize(dims_expander, [299, 299])
    normalized     = tf.divide(tf.subtract(resized, [128]), [128])

with tf.Session(graph=prep_graph) as sess:
    tensor = sess.run(normalized)

print(f"Preprocessed tensor shape: {tensor.shape}")
print(f"Value range: min={tensor.min():.3f}, max={tensor.max():.3f}")
print(f"  (expected: roughly -1.0 to +1.0)")
print()

# ── Run inference ─────────────────────────────────────────────────────────────
input_op  = graph.get_operation_by_name("import/Mul")
output_op = graph.get_operation_by_name("import/final_result")

with tf.Session(graph=graph) as sess:
    scores = sess.run(output_op.outputs[0], {input_op.outputs[0]: tensor})

scores = np.squeeze(scores)
print("="*50)
print("RESULTS:")
print("="*50)
for i, label in enumerate(labels):
    bar = "#" * int(scores[i] * 50)
    marker = " <-- PREDICTED" if i == int(np.argmax(scores)) else ""
    print(f"  {label:<14}: {scores[i]*100:6.2f}%  [{bar}]{marker}")

print()
best_idx   = int(np.argmax(scores))
best_label = labels[best_idx]
best_score = scores[best_idx]
print(f"Final prediction : {best_label.title()}")
print(f"Confidence       : {best_score*100:.1f}%")

if best_score < 0.30:
    print("\nWARNING: Confidence is very low (<30%).")
    print("The image may not be a suitable skin/tissue image,")
    print("or the model may not recognise this type of image.")
elif best_score < 0.60:
    print("\nNOTE: Moderate confidence. Result may not be reliable.")
else:
    print("\nStatus: Good confidence level.")

print()
