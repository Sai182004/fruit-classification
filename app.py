# app.py
import os, time, json, logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ExifTags
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fruit-classifier")

# config (override via ENV if needed)
MODEL_PATH = os.environ.get("MODEL_PATH", "models/fruit_cnn.h5")
LABELS_PATH = os.environ.get("LABELS_PATH", "models/labels.json")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0")
MODEL_ACCURACY = os.environ.get("MODEL_ACCURACY", "99.81%")
TARGET_SIZE = tuple(int(x) for x in os.environ.get("TARGET_SIZE", "224,224").split(","))

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, origins="*")  # demo: allow all origins; lock down in production

# Load labels
if Path(LABELS_PATH).exists():
    with open(LABELS_PATH, "r", encoding="utf-8") as fh:
        LABELS = json.load(fh)
else:
    LABELS = [
        'freshapples', 'freshbanana', 'freshoranges',
        'rottenapples', 'rottenbanana', 'rottenoranges'
    ]

logger.info("Loading model from %s", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
model.make_predict_function()
logger.info("Model loaded. version=%s labels=%s", MODEL_VERSION, LABELS)

def apply_exif_orientation(img: Image.Image) -> Image.Image:
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = img._getexif()
        if exif is None:
            return img
        orientation_val = exif.get(orientation, None)
        if orientation_val == 3:
            img = img.rotate(180, expand=True)
        elif orientation_val == 6:
            img = img.rotate(270, expand=True)
        elif orientation_val == 8:
            img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def preprocess_image(pil_img: Image.Image, target_size=TARGET_SIZE, normalize="0_1"):
    pil_img = apply_exif_orientation(pil_img)
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = np.asarray(pil_img).astype("float32")
    if normalize == "0_1":
        arr = arr / 255.0
    elif normalize == "-1_1":
        arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/")
def index():
    if Path("static/index.html").exists():
        return send_from_directory("static", "index.html")
    return "Fruit classifier API. Use POST /api/predict"

@app.route("/health")
def health():
    return jsonify({"status":"ok","model_version":MODEL_VERSION})

@app.route("/api/predict", methods=["POST"])
def predict_api():
    start = time.time()
    if "file" not in request.files:
        return jsonify({"error":"file field required (multipart/form-data)"}), 400
    f = request.files["file"]
    try:
        img = Image.open(f.stream)
    except Exception as e:
        logger.exception("Invalid image")
        return jsonify({"error":"invalid image", "detail": str(e)}), 400

    # debug info
    try:
        arr_pre = np.asarray(img)
        logger.info("Received file=%s content_type=%s size=%s mode=%s arr_shape=%s dtype=%s min=%s max=%s",
                    getattr(f, "filename", ""), f.content_type, request.content_length, img.mode,
                    arr_pre.shape, arr_pre.dtype, float(arr_pre.min()), float(arr_pre.max()))
    except Exception:
        logger.info("Could not inspect raw image array")

    X = preprocess_image(img, target_size=TARGET_SIZE, normalize="0_1")

    try:
        preds = model.predict(X)
    except Exception as e:
        logger.exception("Model inference error")
        return jsonify({"error":"model inference error", "detail": str(e)}), 500

    scores = preds[0].tolist()
    sorted_idx = np.argsort(preds[0])[::-1]
    top_k = int(request.form.get("top_k", 3))
    top_k = min(top_k, len(scores))
    top = []
    for i in range(top_k):
        idx = int(sorted_idx[i])
        label = LABELS[idx] if idx < len(LABELS) else str(idx)
        score = float(scores[idx])
        top.append({"index": idx, "label": label, "score": score})

    best_idx = int(sorted_idx[0])
    best_label = LABELS[best_idx] if best_idx < len(LABELS) else str(best_idx)
    confidence = float(scores[best_idx]) * 100.0

    elapsed = time.time() - start
    response = {
        "model_version": MODEL_VERSION,
        "prediction": best_label,
        "confidence_score": f"{confidence:.2f}%",
        "model_accuracy": MODEL_ACCURACY,
        "top_k": top,
        "raw_scores": scores,
        "timing": {"inference_seconds": round(elapsed, 4)}
    }
    logger.info("Predicted %s (%.2f%%) in %.3fs", best_label, confidence, elapsed)
    return jsonify(response)

@app.route("/api/feedback", methods=["POST"])
def feedback():
    os.makedirs("uploads/feedback", exist_ok=True)
    if "file" not in request.files:
        return jsonify({"error":"file required"}), 400
    f = request.files["file"]
    predicted = request.form.get("predicted","")
    correct = request.form.get("correct","")
    tag = request.form.get("tag","")
    fname = f"{int(time.time())}_{f.filename}"
    save_path = os.path.join("uploads","feedback", fname)
    f.save(save_path)
    meta = {"filename":fname,"predicted":predicted,"correct":correct,"tag":tag}
    with open(save_path + ".json","w",encoding="utf-8") as fh:
        json.dump(meta, fh)
    return jsonify({"status":"saved","path":save_path})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
