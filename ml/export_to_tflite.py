"""
Export trained YOLO .pt model to TFLite format for Android deployment.
Since ultralytics 8.4.x on Windows doesn't support direct TFLite export,
we go: .pt -> ONNX (via ultralytics) -> TFLite (via onnx2tf)
"""
from pathlib import Path
import shutil
import sys

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR.parent / "android-app" / "app" / "src" / "main" / "assets"

# Find the latest .pt file
pt_files = sorted(OUTPUTS_DIR.glob("*.pt"))
if not pt_files:
    print("ERROR: No .pt file found in ml/outputs/")
    sys.exit(1)

PT_MODEL = pt_files[-1]
print(f"[1/4] Using PT model: {PT_MODEL.name}")

# -------- Step 1: Export to ONNX via ultralytics --------
onnx_files = sorted(OUTPUTS_DIR.glob("*.onnx"))
if onnx_files:
    ONNX_MODEL = onnx_files[-1]
    print(f"\n[1/4] Reusing existing ONNX: {ONNX_MODEL.name}")
else:
    print("\n[1/4] Exporting .pt -> ONNX ...")
    from ultralytics import YOLO
    model = YOLO(str(PT_MODEL))
    onnx_path_str = model.export(format="onnx", imgsz=640, opset=17, simplify=True)
    ONNX_MODEL = Path(onnx_path_str)
    # Also save model names for later
    class_names = list(model.names.values()) if hasattr(model, "names") else None
    (OUTPUTS_DIR / "_class_names.txt").write_text("\n".join(class_names or []), encoding="utf-8")
    print(f"  ONNX saved: {ONNX_MODEL}")

# -------- Step 2: Convert ONNX -> TFLite via onnx2tf CLI --------
print("\n[2/4] Converting ONNX -> TFLite ...")
import subprocess
env = None
tflite_out_dir = OUTPUTS_DIR / "tflite_converted"
if tflite_out_dir.exists():
    shutil.rmtree(tflite_out_dir)
# Use onnx2tf directly via subprocess - it handles its own TF context better
result = subprocess.run(
    [
        sys.executable, "-m", "onnx2tf",
        "-i", str(ONNX_MODEL),
        "-o", str(tflite_out_dir),
        "-osd",  # output structure in detail
    ],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(result.stdout[-2000:])
    print(result.stderr[-2000:])
    print("ERROR: onnx2tf conversion failed")
    sys.exit(1)
# Find the generated .tflite
tflite_candidates = list(tflite_out_dir.rglob("*.tflite"))
if not tflite_candidates:
    print(f"ERROR: No .tflite found under {tflite_out_dir}")
    print("\nDirectory listing:")
    for p in sorted(tflite_out_dir.rglob("*")):
        print(" ", p.relative_to(tflite_out_dir))
    sys.exit(1)
TFLITE_SRC = tflite_candidates[0]
TFLITE_OUTPUT = OUTPUTS_DIR / "detect.tflite"
shutil.copy2(TFLITE_SRC, TFLITE_OUTPUT)
print(f"  TFLite: {TFLITE_OUTPUT} ({TFLITE_OUTPUT.stat().st_size/1024/1024:.1f} MB)")

# -------- Step 3: Prepare labels.txt --------
print("\n[3/4] Preparing labels.txt ...")
class_names_path = OUTPUTS_DIR / "_class_names.txt"
if class_names_path.exists():
    class_names = [l.strip() for l in class_names_path.read_text(encoding="utf-8").splitlines() if l.strip()]
else:
    # Read from beibingyang.yaml (names section)
    import re
    yaml_text = (BASE_DIR / "beibingyang.yaml").read_text(encoding="utf-8")
    # Extract class names under names: in order
    class_names = []
    in_names = False
    for line in yaml_text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            m = re.match(r"\s*(\d+):\s*(.+)", stripped)
            if m:
                class_names.append(m.group(2).strip())
            elif stripped and not stripped.startswith(" "):
                break
    if not class_names:
        class_names = ["beibingyang_can"]
LABELS_OUTPUT = OUTPUTS_DIR / "labels.txt"
LABELS_OUTPUT.write_text("\n".join(class_names) + "\n", encoding="utf-8")
print(f"  Labels ({len(class_names)}): {class_names}")

# -------- Step 4: Copy to Android assets --------
print("\n[4/4] Copying to Android assets directory ...")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
dest_tflite = ASSETS_DIR / "detect.tflite"
dest_labels = ASSETS_DIR / "labels.txt"
shutil.copy2(TFLITE_OUTPUT, dest_tflite)
shutil.copy2(LABELS_OUTPUT, dest_labels)
print(f"  -> {dest_tflite}")
print(f"  -> {dest_labels}")
print("\nDone! Rebuild the Android app (gradlew assembleDebug) to package the new model.")
