import os
import sys
import json
import base64
import shutil
import tempfile
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

import cv2
import yaml
from flask import Flask, request, jsonify, send_file, Response
from ultralytics import YOLOWorld, YOLO

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "_uploads"
DATASET_DIR = BASE_DIR / "_dataset"
OUTPUT_DIR = BASE_DIR / "_outputs"
TMP_DIR = BASE_DIR / "_tmp"

for d in [UPLOAD_DIR, DATASET_DIR, OUTPUT_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

state = {
    "status": "idle",
    "step": "",
    "progress": 0,
    "message": "",
    "logs": [],
    "error": None,
    "result": None,
}
state_lock = threading.Lock()

yolo_world_model = None
LLM_BASE_URL = "http://127.0.0.1:12345"
LLM_MODEL = "qwen3.5"


def llm_available():
    try:
        req = urllib.request.Request(f"{LLM_BASE_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return True
    except Exception:
        return False


def llm_chat(messages, temperature=0.3, max_tokens=1024):
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def generate_llm_prompts(class_name):
    try:
        resp = llm_chat([
            {"role": "system", "content": "You are a computer vision expert. Generate diverse text prompts for detecting objects in images."},
            {"role": "user", "content": (
                f"Generate 5-8 short text prompts (single words or brief phrases) for detecting a '{class_name}' "
                f"in images. Include variations in naming, materials, shapes, and contexts. "
                f"Return ONLY a comma-separated list of prompts, no explanation, no numbering."
            )},
        ], temperature=0.7, max_tokens=128)
        prompts = [p.strip().rstrip(".") for p in resp.split(",") if p.strip()]
        if prompts:
            return list(dict.fromkeys(prompts[:8]))
    except Exception:
        pass
    return generate_prompts(class_name)


def llm_analyze_image(image_path, class_name):
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
        b64 = base64.b64encode(img_data).decode("utf-8")

        prompt = (
            f"Look at this image. Is there a '{class_name}' in the image? "
            f"If yes, describe its appearance (color, shape, material, position). "
            f"If no, say 'not present'. Keep answer under 50 words."
        )
        resp = llm_chat([
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ], temperature=0.1, max_tokens=256)
        return resp.strip()
    except Exception:
        return None


def update_status(step, progress, message="", log=None):
    with state_lock:
        state["status"] = "running"
        state["step"] = step
        state["progress"] = progress
        state["message"] = message
        if log:
            state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {log}")
        state["error"] = None


def set_error(message):
    with state_lock:
        state["status"] = "error"
        state["error"] = message
        state["message"] = message
        state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {message}")


def set_done(result=None):
    with state_lock:
        state["status"] = "done"
        state["progress"] = 100
        state["step"] = "complete"
        state["message"] = "Pipeline completed successfully!"
        if result:
            state["result"] = result
        state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Pipeline completed successfully!")


def get_status():
    with state_lock:
        return dict(state)


def reset_state():
    with state_lock:
        state["status"] = "idle"
        state["step"] = ""
        state["progress"] = 0
        state["message"] = ""
        state["logs"] = []
        state["error"] = None
        state["result"] = None


def extract_frames(video_path, output_dir, fps=2):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = max(1, int(round(video_fps / fps))) if video_fps > 0 else 1
    extracted = 0
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            name = f"frame_{saved_idx + 1:04d}"
            cv2.imwrite(str(output_dir / f"{name}.jpg"), frame)
            extracted += 1
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return extracted, width, height


def generate_prompts(class_name):
    name = class_name.lower().replace("_", " ").replace("-", " ").strip()
    words = name.split()
    prompts = [class_name, name]
    if len(words) >= 2:
        prompts.append(" ".join(words[-1:]))
        prompts.append(" ".join(words[:1]))
    generic = ["object", "item", "thing"]
    for g in generic:
        prompts.append(f"{name} {g}")
    return list(dict.fromkeys(prompts))


def auto_label_frames(frames_dir, labels_dir, class_name, conf_threshold=0.25, use_llm=False):
    global yolo_world_model
    if yolo_world_model is None:
        update_status("init_model", 5, "Loading YOLO-World model...", "Loading YOLO-World model for auto-labeling")
        yolo_world_model = YOLOWorld("yolov8s-worldv2.pt")

    if use_llm and llm_available():
        update_status("init_model", 10, "LLM available, generating enhanced prompts...", "Using LLM to generate enhanced detection prompts")
        prompts = generate_llm_prompts(class_name)
        update_status("init_model", 15, f"Generated {len(prompts)} prompts", f"LLM prompts: {', '.join(prompts)}")
    else:
        prompts = generate_prompts(class_name)
        if use_llm:
            update_status("init_model", 10, "LLM not available, using default prompts", "Falling back to default prompt generation")

    yolo_world_model.set_classes(prompts)

    image_files = sorted(frames_dir.glob("*.jpg"))
    total = len(image_files)
    labeled = 0
    llm_verified = 0

    for idx, img_path in enumerate(image_files):
        results = yolo_world_model.predict(
            str(img_path), save=False, verbose=False, conf=conf_threshold
        )
        label_path = labels_dir / f"{img_path.stem}.txt"

        has_detection = False
        boxes_to_write = []

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for i in range(len(r.boxes)):
                    conf = r.boxes.conf[i].item()
                    xyxyn = r.boxes.xyxyn[i].tolist()
                    cx = (xyxyn[0] + xyxyn[2]) / 2.0
                    cy = (xyxyn[1] + xyxyn[3]) / 2.0
                    bw = xyxyn[2] - xyxyn[0]
                    bh = xyxyn[3] - xyxyn[1]

                    if use_llm and llm_available() and conf < 0.4 and labeled > 0:
                        analysis = llm_analyze_image(img_path, class_name)
                        if analysis and "not present" not in analysis.lower():
                            boxes_to_write.append((cx, cy, bw, bh))
                            llm_verified += 1
                        elif analysis and "not present" in analysis.lower():
                            pass
                        else:
                            boxes_to_write.append((cx, cy, bw, bh))
                    else:
                        boxes_to_write.append((cx, cy, bw, bh))

        if boxes_to_write:
            with open(label_path, "w") as f:
                for cx, cy, bw, bh in boxes_to_write:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            has_detection = True
        else:
            label_path.write_text("")

        if has_detection:
            labeled += 1

        progress = int((idx + 1) / total * 100)
        log_msg = f"{'Labeled' if has_detection else 'Skipped'}: {img_path.name}"
        if use_llm and llm_available():
            log_msg += f" (LLM verified: {llm_verified})"
        update_status(
            "labeling",
            progress,
            f"Labeling frame {idx + 1}/{total}",
            log_msg,
        )

    return labeled


def split_dataset(images_dir, labels_dir, train_ratio=0.9):
    image_files = sorted(images_dir.glob("*.jpg"))
    total = len(image_files)
    split_idx = int(total * train_ratio)

    train_images = images_dir / "train"
    val_images = images_dir / "val"
    train_labels = labels_dir / "train"
    val_labels = labels_dir / "val"

    for d in [train_images, val_images, train_labels, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_files):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if i < split_idx:
            dest_img = train_images / img_path.name
            dest_lbl = train_labels / lbl_path.name
        else:
            dest_img = val_images / img_path.name
            dest_lbl = val_labels / lbl_path.name

        shutil.copy2(str(img_path), str(dest_img))
        if lbl_path.exists():
            shutil.copy2(str(lbl_path), str(dest_lbl))


def generate_yaml(dataset_dir, class_name, yaml_path):
    config = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def run_pipeline(video_path, class_name, config):
    try:
        reset_state()
        video_path = Path(video_path)

        fps = config.get("fps", 2)
        epochs = config.get("epochs", 100)
        imgsz = config.get("imgsz", 640)
        batch = config.get("batch", 8)
        device = config.get("device", "0")
        workers = config.get("workers", 0)
        conf_threshold = config.get("conf_threshold", 0.25)
        train_ratio = config.get("train_ratio", 0.9)
        use_llm = config.get("use_llm", False)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = DATASET_DIR / run_id

        frames_dir = run_dir / "frames"
        labels_dir = run_dir / "labels"
        images_dir = run_dir / "images"
        dataset_dir = run_dir / "dataset"

        for d in [frames_dir, labels_dir, images_dir, dataset_dir]:
            d.mkdir(parents=True, exist_ok=True)

        update_status("extracting", 0, "Extracting frames from video...", f"Starting frame extraction (fps={fps})")
        frame_count, width, height = extract_frames(video_path, frames_dir, fps=fps)
        update_status("extracting", 100, f"Extracted {frame_count} frames", f"Extracted {frame_count} frames ({width}x{height})")

        update_status("labeling", 0, "Auto-labeling frames with YOLO-World...", "Starting auto-labeling with YOLO-World")
        labeled_count = auto_label_frames(frames_dir, labels_dir, class_name, conf_threshold=conf_threshold, use_llm=use_llm)
        update_status("labeling", 100, f"Labeled {labeled_count}/{frame_count} frames", f"Labeling complete: {labeled_count}/{frame_count} frames with detections")

        update_status("splitting", 0, "Splitting dataset into train/val...", f"Train ratio: {train_ratio}")
        split_dataset(frames_dir, labels_dir, train_ratio=train_ratio)

        for d in [dataset_dir / "images", dataset_dir / "labels"]:
            if d.exists():
                shutil.rmtree(str(d))

        shutil.copytree(str(frames_dir.parent / "train"), str(dataset_dir / "images" / "train"))
        shutil.copytree(str(frames_dir.parent / "val"), str(dataset_dir / "images" / "val"))
        shutil.copytree(str(labels_dir.parent / "train"), str(dataset_dir / "labels" / "train"))
        shutil.copytree(str(labels_dir.parent / "val"), str(dataset_dir / "labels" / "val"))

        yaml_path = run_dir / "dataset.yaml"
        generate_yaml(dataset_dir, class_name, yaml_path)
        update_status("splitting", 100, "Dataset split complete", f"YAML config: {yaml_path}")

        update_status("training", 0, f"Training YOLO model (epochs={epochs})...", "Starting YOLO training")

        model = YOLO("yolo11n.pt")

        progress_callback = []

        class ProgressTracker:
            def __init__(self):
                self.last_pct = -1

            def __call__(self, trainer):
                try:
                    pct = int(trainer.epoch / trainer.epochs * 100)
                    if pct != self.last_pct:
                        self.last_pct = pct
                        update_status(
                            "training",
                            pct,
                            f"Training epoch {trainer.epoch}/{trainer.epochs}",
                            f"Epoch {trainer.epoch}/{trainer.epochs}",
                        )
                except Exception:
                    pass

        tracker = ProgressTracker()

        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=str(OUTPUT_DIR),
            name=run_id,
            callbacks={"on_epoch_end": tracker},
        )

        best_model_path = OUTPUT_DIR / run_id / "weights" / "best.pt"
        if not best_model_path.exists():
            best_model_path = OUTPUT_DIR / run_id / "weights" / "last.pt"

        final_model_path = OUTPUT_DIR / f"{class_name}_{run_id}.pt"
        shutil.copy2(str(best_model_path), str(final_model_path))

        update_status("done", 100, "Training complete!", f"Model saved to: {final_model_path}")

        set_done({
            "model_path": str(final_model_path),
            "model_name": final_model_path.name,
            "run_id": run_id,
            "frames": frame_count,
            "labeled": labeled_count,
            "class_name": class_name,
        })

    except Exception as e:
        set_error(str(e))
        import traceback
        traceback.print_exc()


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/status")
def api_status():
    return jsonify(get_status())


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format: {ext}. Allowed: {', '.join(allowed)}"}), 400

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = UPLOAD_DIR / f"{run_id}{ext}"
    f.save(str(video_path))

    return jsonify({"video_path": str(video_path), "filename": f.filename})


@app.route("/api/process", methods=["POST"])
def api_process():
    data = request.get_json()
    video_path = data.get("video_path")
    class_name = data.get("class_name", "object")

    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Video file not found"}), 400

    class_name = class_name.strip()
    if not class_name:
        return jsonify({"error": "Class name is required"}), 400

    config = {
        "fps": data.get("fps", 2),
        "epochs": data.get("epochs", 100),
        "imgsz": data.get("imgsz", 640),
        "batch": data.get("batch", 8),
        "device": data.get("device", "0"),
        "workers": data.get("workers", 0),
        "conf_threshold": data.get("conf_threshold", 0.25),
        "train_ratio": data.get("train_ratio", 0.9),
        "use_llm": data.get("use_llm", False),
    }

    t = threading.Thread(target=run_pipeline, args=(video_path, class_name, config), daemon=True)
    t.start()

    return jsonify({"status": "started"})


@app.route("/api/download/<path:filename>")
def api_download(filename):
    safe_name = os.path.basename(filename)
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(file_path), as_attachment=True, download_name=safe_name)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    reset_state()
    return jsonify({"status": "reset"})


@app.route("/api/llm_status")
def api_llm_status():
    available = llm_available()
    return jsonify({
        "available": available,
        "url": LLM_BASE_URL if available else None,
        "model": LLM_MODEL if available else None,
    })


HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YOLO 视频训练平台</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
    color: #e0e0e0;
    padding: 20px;
  }
  .container { max-width: 960px; margin: 0 auto; }
  h1 {
    text-align: center;
    font-size: 2rem;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .subtitle {
    text-align: center;
    color: #999;
    margin-bottom: 30px;
    font-size: 0.95rem;
  }
  .card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
  }
  .card h2 { font-size: 1.1rem; margin-bottom: 16px; color: #b8b8ff; }

  .upload-area {
    border: 2px dashed rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    background: rgba(255,255,255,0.02);
  }
  .upload-area:hover, .upload-area.dragover {
    border-color: #667eea;
    background: rgba(102,126,234,0.1);
  }
  .upload-area .icon { font-size: 3rem; margin-bottom: 12px; }
  .upload-area p { color: #aaa; }
  .upload-area .filename { color: #667eea; font-weight: 500; margin-top: 8px; }
  input[type="file"] { display: none; }

  .form-group { margin-bottom: 16px; }
  .form-group label { display: block; margin-bottom: 6px; color: #bbb; font-size: 0.9rem; }
  .form-group input, .form-group select {
    width: 100%;
    padding: 10px 14px;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    color: #fff;
    font-size: 0.95rem;
    transition: border-color 0.2s;
  }
  .form-group input:focus, .form-group select:focus {
    outline: none;
    border-color: #667eea;
  }
  .form-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }

  .btn {
    display: inline-block;
    padding: 12px 28px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
  }
  .btn-primary {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
  }
  .btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102,126,234,0.4); }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-secondary {
    background: rgba(255,255,255,0.1);
    color: #ddd;
    border: 1px solid rgba(255,255,255,0.2);
  }
  .btn-secondary:hover:not(:disabled) { background: rgba(255,255,255,0.15); }
  .btn-group { display: flex; gap: 12px; margin-top: 20px; }

  .progress-section { display: none; }
  .progress-section.active { display: block; }
  .progress-bar {
    width: 100%;
    height: 8px;
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
    overflow: hidden;
    margin: 12px 0;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 4px;
    transition: width 0.3s;
    width: 0%;
  }
  .progress-info { display: flex; justify-content: space-between; font-size: 0.85rem; color: #aaa; }
  .step-indicator {
    display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap;
  }
  .step {
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    background: rgba(255,255,255,0.05);
    color: #888;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .step.active { background: rgba(102,126,234,0.3); color: #fff; border-color: #667eea; }
  .step.done { background: rgba(76,175,80,0.2); color: #81c784; border-color: #4caf50; }

  .log-area {
    background: rgba(0,0,0,0.4);
    border-radius: 8px;
    padding: 14px;
    font-family: "Consolas", "Monaco", monospace;
    font-size: 0.8rem;
    max-height: 250px;
    overflow-y: auto;
    margin-top: 12px;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .log-area .log-line { color: #90caf9; margin-bottom: 2px; }
  .log-area .log-line.error { color: #ef5350; }
  .log-area .log-line.success { color: #66bb6a; }

  .result-section { display: none; }
  .result-section.active { display: block; }
  .result-box {
    background: rgba(76,175,80,0.1);
    border: 1px solid rgba(76,175,80,0.3);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
  }
  .result-box .icon { font-size: 3rem; margin-bottom: 12px; }
  .result-box h3 { color: #81c784; margin-bottom: 8px; }
  .result-box p { color: #aaa; margin-bottom: 16px; }
  .download-btn {
    display: inline-block;
    padding: 12px 32px;
    background: linear-gradient(90deg, #4caf50, #2e7d32);
    color: white;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 600;
    transition: all 0.2s;
  }
  .download-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(76,175,80,0.4); }

  .settings-toggle {
    display: flex; align-items: center; gap: 8px;
    cursor: pointer; color: #999; font-size: 0.85rem;
    user-select: none;
  }
  .settings-toggle:hover { color: #ccc; }
  .settings-content { display: none; margin-top: 16px; }
  .settings-content.open { display: block; }

  .status-badge {
    display: inline-block; padding: 4px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600;
  }
  .status-idle { background: rgba(255,255,255,0.1); color: #aaa; }
  .status-running { background: rgba(102,126,234,0.3); color: #b8b8ff; }
  .status-done { background: rgba(76,175,80,0.3); color: #81c784; }
  .status-error { background: rgba(244,67,54,0.3); color: #ef5350; }
</style>
</head>
<body>
<div class="container">
  <h1>🎯 YOLO 视频训练平台</h1>
  <p class="subtitle">上传视频 → 自动标注 → 训练模型 → 输出 .pt</p>

  <div class="card">
    <h2>1. 上传视频</h2>
    <div class="upload-area" id="uploadArea">
      <div class="icon">📹</div>
      <p>点击选择视频文件，或拖拽到此处</p>
      <p style="font-size:0.8rem;color:#666;margin-top:8px;">支持 MP4, AVI, MOV, MKV, WEBM 等格式</p>
      <div class="filename" id="fileName"></div>
    </div>
    <input type="file" id="videoFile" accept="video/*">
  </div>

  <div class="card">
    <h2>2. 配置识别目标</h2>
    <div class="form-group">
      <label>识别目标（用英文，如 beibingyang_can、bottle、red_car）</label>
      <input type="text" id="className" placeholder="例如：beibingyang_can" value="beibingyang_can">
    </div>

    <div class="settings-toggle" onclick="toggleSettings()">
      <span>⚙️ 高级设置（点击展开）</span>
    </div>
    <div class="settings-content" id="settingsContent">
      <div class="form-row">
        <div class="form-group">
          <label>抽帧频率 (FPS)</label>
          <input type="number" id="fps" value="2" min="0.5" max="30" step="0.5">
        </div>
        <div class="form-group">
          <label>置信度阈值</label>
          <input type="number" id="confThreshold" value="0.25" min="0.05" max="0.95" step="0.05">
        </div>
        <div class="form-group">
          <label>训练/验证比例</label>
          <input type="number" id="trainRatio" value="0.9" min="0.5" max="0.95" step="0.05">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>训练轮数 (Epochs)</label>
          <input type="number" id="epochs" value="100" min="1" max="1000">
        </div>
        <div class="form-group">
          <label>图像尺寸</label>
          <select id="imgsz">
            <option value="640" selected>640</option>
            <option value="480">480</option>
            <option value="320">320</option>
            <option value="1280">1280</option>
          </select>
        </div>
        <div class="form-group">
          <label>批次大小 (Batch)</label>
          <input type="number" id="batch" value="8" min="1" max="128">
        </div>
        <div class="form-group">
          <label>设备</label>
          <select id="device">
            <option value="0" selected>GPU (0)</option>
            <option value="cpu">CPU</option>
            <option value="0,1">GPU (0,1)</option>
          </select>
        </div>
      </div>
      <div class="form-group">
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;">
          <input type="checkbox" id="useLLM" style="width:auto;">
          <span>使用本地 LLM 辅助标注（qwen3.5 - 增强提示词 + 低置信度复核）</span>
          <span id="llmStatus" style="font-size:0.75rem;padding:2px 8px;border-radius:10px;background:rgba(255,255,255,0.1);color:#888;"></span>
        </label>
      </div>
    </div>

    <div class="btn-group">
      <button class="btn btn-primary" id="startBtn" onclick="startPipeline()">🚀 开始训练</button>
      <button class="btn btn-secondary" id="resetBtn" onclick="resetPipeline()">🔄 重置</button>
    </div>
  </div>

  <div class="card progress-section" id="progressSection">
    <h2>
      3. 训练进度
      <span class="status-badge status-idle" id="statusBadge">idle</span>
    </h2>
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <div class="progress-info">
      <span id="progressStep">等待开始...</span>
      <span id="progressPct">0%</span>
    </div>
    <div class="step-indicator" id="stepIndicator">
      <div class="step" data-step="extracting">📸 抽帧</div>
      <div class="step" data-step="labeling">🏷️ 标注</div>
      <div class="step" data-step="splitting">📊 分割</div>
      <div class="step" data-step="training">🧠 训练</div>
    </div>
    <div class="log-area" id="logArea"></div>
  </div>

  <div class="card result-section" id="resultSection">
    <div class="result-box">
      <div class="icon">🎉</div>
      <h3>训练完成！</h3>
      <p id="resultInfo"></p>
      <a class="download-btn" id="downloadBtn" href="#">⬇️ 下载模型 (.pt)</a>
    </div>
  </div>
</div>

<script>
let videoPath = null;
let statusPoller = null;

const uploadArea = document.getElementById('uploadArea');
const videoFile = document.getElementById('videoFile');
const fileName = document.getElementById('fileName');

uploadArea.addEventListener('click', () => videoFile.click());
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    videoFile.files = e.dataTransfer.files;
    handleFile();
  }
});
videoFile.addEventListener('change', handleFile);

checkLLMStatus();

async function checkLLMStatus() {
  try {
    const resp = await fetch('/api/llm_status');
    const data = await resp.json();
    const badge = document.getElementById('llmStatus');
    if (data.available) {
      badge.textContent = '● 已连接';
      badge.style.background = 'rgba(76,175,80,0.2)';
      badge.style.color = '#81c784';
    } else {
      badge.textContent = '○ 未连接';
      badge.style.background = 'rgba(244,67,54,0.15)';
      badge.style.color = '#ef5350';
    }
  } catch(e) {
    const badge = document.getElementById('llmStatus');
    badge.textContent = '○ 未连接';
    badge.style.background = 'rgba(244,67,54,0.15)';
    badge.style.color = '#ef5350';
  }
}

function handleFile() {
  const f = videoFile.files[0];
  if (f) {
    fileName.textContent = `📎 ${f.name} (${(f.size / 1024 / 1024).toFixed(1)} MB)`;
    uploadArea.style.borderColor = '#4caf50';
  }
}

function toggleSettings() {
  document.getElementById('settingsContent').classList.toggle('open');
}

function setStatusBadge(status) {
  const badge = document.getElementById('statusBadge');
  badge.className = 'status-badge status-' + status;
  badge.textContent = status;
}

function updateStepIndicator(activeStep) {
  document.querySelectorAll('.step').forEach(el => {
    const step = el.dataset.step;
    el.classList.remove('active', 'done');
    const stepOrder = ['extracting', 'init_model', 'labeling', 'splitting', 'training'];
    const activeIdx = stepOrder.indexOf(activeStep);
    const elIdx = stepOrder.indexOf(step);
    if (activeIdx >= 0 && elIdx < activeIdx) el.classList.add('done');
    else if (step === activeStep) el.classList.add('active');
  });
}

async function startPipeline() {
  if (!videoFile.files.length) {
    alert('请先上传视频文件');
    return;
  }
  const className = document.getElementById('className').value.trim();
  if (!className) {
    alert('请填写识别目标名称');
    return;
  }

  const btn = document.getElementById('startBtn');
  btn.disabled = true;
  btn.textContent = '⏳ 处理中...';

  const formData = new FormData();
  formData.append('video', videoFile.files[0]);

  try {
    const uploadResp = await fetch('/api/upload', { method: 'POST', body: formData });
    const uploadData = await uploadResp.json();
    if (uploadData.error) throw new Error(uploadData.error);
    videoPath = uploadData.video_path;

    document.getElementById('progressSection').classList.add('active');
    document.getElementById('logArea').innerHTML = '';

    const processResp = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_path: videoPath,
        class_name: className,
        fps: parseFloat(document.getElementById('fps').value),
        epochs: parseInt(document.getElementById('epochs').value),
        imgsz: parseInt(document.getElementById('imgsz').value),
        batch: parseInt(document.getElementById('batch').value),
        device: document.getElementById('device').value,
        workers: 0,
        conf_threshold: parseFloat(document.getElementById('confThreshold').value),
        train_ratio: parseFloat(document.getElementById('trainRatio').value),
        use_llm: document.getElementById('useLLM').checked,
      })
    });

    if (processResp.status === 200) {
      startPolling();
    } else {
      const err = await processResp.json();
      throw new Error(err.error || 'Failed to start');
    }
  } catch (e) {
    alert('错误: ' + e.message);
    resetPipeline();
  }
}

function startPolling() {
  if (statusPoller) clearInterval(statusPoller);
  statusPoller = setInterval(async () => {
    try {
      const resp = await fetch('/api/status');
      const data = await resp.json();
      updateUI(data);
      if (data.status === 'done' || data.status === 'error') {
        clearInterval(statusPoller);
        statusPoller = null;
      }
    } catch (e) {
      console.error('Status poll error:', e);
    }
  }, 500);
}

function updateUI(data) {
  document.getElementById('progressFill').style.width = data.progress + '%';
  document.getElementById('progressPct').textContent = data.progress + '%';
  document.getElementById('progressStep').textContent = data.message || data.step;
  setStatusBadge(data.status);
  updateStepIndicator(data.step);

  const logArea = document.getElementById('logArea');
  if (data.logs && data.logs.length) {
    logArea.innerHTML = data.logs.map(l => {
      const cls = data.status === 'error' ? 'error' : '';
      return `<div class="log-line ${cls}">${l}</div>`;
    }).join('');
    logArea.scrollTop = logArea.scrollHeight;
  }

  if (data.status === 'done' && data.result) {
    document.getElementById('resultSection').classList.add('active');
    document.getElementById('resultInfo').innerHTML =
      `类别: <b>${data.result.class_name}</b> | 抽帧: ${data.result.frames} | 标注: ${data.result.labeled}`;
    document.getElementById('downloadBtn').href = '/api/download/' + encodeURIComponent(data.result.model_name);
    document.getElementById('startBtn').disabled = false;
    document.getElementById('startBtn').textContent = '🚀 重新训练';
  }

  if (data.status === 'error') {
    document.getElementById('startBtn').disabled = false;
    document.getElementById('startBtn').textContent = '🚀 开始训练';
  }
}

async function resetPipeline() {
  if (statusPoller) { clearInterval(statusPoller); statusPoller = null; }
  try { await fetch('/api/reset', { method: 'POST' }); } catch(e) {}
  resetUI();
}

function resetUI() {
  videoPath = null;
  videoFile.value = '';
  fileName.textContent = '';
  uploadArea.style.borderColor = '';
  document.getElementById('progressSection').classList.remove('active');
  document.getElementById('resultSection').classList.remove('active');
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressPct').textContent = '0%';
  document.getElementById('progressStep').textContent = '等待开始...';
  document.getElementById('logArea').innerHTML = '';
  setStatusBadge('idle');
  document.querySelectorAll('.step').forEach(el => el.classList.remove('active', 'done'));
  document.getElementById('startBtn').disabled = false;
  document.getElementById('startBtn').textContent = '🚀 开始训练';
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  YOLO 视频训练平台 启动中...")
    print("  访问地址: http://127.0.0.1:5000")
    print("  按 Ctrl+C 停止服务")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
