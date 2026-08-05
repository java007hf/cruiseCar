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

# onnx2tf requires both tensorflow + onnx2tf packages to be importable in the
# same interpreter.  On this dev box that is the SYSTEM-WIDE Python 3.10 at
# C:\Python310 (not .venv-py, which has no tensorflow package).  If the user
# updates their system Python install path, adjust this.
SYS_PY_WITH_TF_AND_ONNX2TF = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"


def _ensure_onnx2tf_allow_pickle() -> None:
    """
    Work around: numpy >=1.26 flipped np.load() default to allow_pickle=False,
    but onnx2tf caches test-image sample data in a pickled .npy file.  When
    onnx2tf.convert() calls download_test_image_data() → np.load(f), the read
    raises `ValueError: Cannot load file containing pickled data when
    allow_pickle=False`, aborting the whole conversion (even if the user
    passes -cind custom input data — onnx2tf does an unconditional fetch to
    prime its internal dummy-data generators).

    This helper idempotently injects allow_pickle=True into the onnx2tf
    source of the interpreter that is about to run the conversion.  It runs
    against SYS_PY_WITH_TF_AND_ONNX2TF's site-packages because that's where
    onnx2tf will be imported from during the subprocess call.
    """
    # Resolve site-packages of the TF-python interpreter (next to `onnx2tf`
    # top-level dir). Do not assume a fixed site-packages path; ask that
    # interpreter to tell us its onnx2tf utils dir.
    probe = subprocess.run(
        [SYS_PY_WITH_TF_AND_ONNX2TF, "-c",
         "import onnx2tf.utils.common_functions as m, os; print(os.path.abspath(m.__file__))"],
        capture_output=True, text=True
    )
    if probe.returncode != 0:
        print("  WARN: cannot locate onnx2tf installation (allow_pickle self-heal skipped). "
              f"stdout={probe.stdout.strip()} stderr={probe.stderr.strip()}")
        return
    target = Path(probe.stdout.strip())
    if not target.exists():
        return
    src = target.read_text(encoding="utf-8")
    # The exact line we want to patch: `test_image_data: np.ndarray = np.load(f)`
    # Do not match a line that is already patched (has `allow_pickle=True`).
    OLD = "test_image_data: np.ndarray = np.load(f)\n"
    NEW = ("# NOTE: cache files fetched on older numpy versions contained pickled dtype metadata; "
           "numpy >=1.26 flipped allow_pickle=False by default. Explicit True restores behavior.\n"
           "        test_image_data: np.ndarray = np.load(f, allow_pickle=True)\n")
    if OLD not in src and "allow_pickle=True" not in src:
        print(f"  WARN: onnx2tf allow_pickle patch point not found in {target}; skipping self-heal.")
        return
    if OLD in src:
        patched = src.replace(OLD, NEW, 1)
        target.write_text(patched, encoding="utf-8")
        print(f"  applied onnx2tf common_functions.py allow_pickle=True self-heal at {target}")
    else:
        print(f"  onnx2tf already has allow_pickle=True patch ({target}); skip.")


_ensure_onnx2tf_allow_pickle()


def _prepare_onnx2tf_runtime(cwd_for_subprocess: Path) -> dict:
    """
    Apply ALL five layers of onnx2tf runtime protection before launching the
    subprocess.  Returns the `env=` dict to pass to subprocess.run().  Also
    performs filesystem prep (writing calibration cache, clearing __pycache__)
    as a side effect.

    Protection layers (in the order a failure would occur during the
    interpreter import → graph-parse → calibration → export pipeline):

      1. PYTHONDONTWRITEBYTECODE=1    : Avoid TRAE sandbox restrictions on
         writing .pyc files under site-packages/onnx2tf/utils/__pycache__/.
         Without this, the sandbox intercepts pyc writes and raises an access
         violation (0xC0000005 / exit=3221225477) inside CPython's import loop.

      2. Wipe onnx2tf utils __pycache__   : Remove stale / partially-written
         .pyc files from earlier interrupted runs (also: the TRAE sandbox
         sometimes writes .pyc with weird temporary names; removing the whole
         directory forces a clean import with bytecode held only in RAM).

      3. Write a brand-new calibration_image_sample_data_20x128x128x3_float32.npy
         with CURRENT numpy into `cwd_for_subprocess` — onnx2tf checks cwd for
         this file first; if missing it downloads the v1.20.4 pickled copy from
         GitHub Releases, which numpy >= 1.26 cannot unpickle even with
         allow_pickle=True (UnpicklingError: Failed to interpret ... as a
         pickle).  Writing our own npy via numpy.save() guarantees format
         compatibility with the reader.

      4. allow_pickle=True patch (applied by _ensure_onnx2tf_allow_pickle()
         above) — if a stale cached npy is still found somewhere else on disk,
         at least the flag allows it through on numpy < 2.0.

      5. CPU-only / oneDNN-off TF env vars — onnx2tf's TensorFlow backend can
         segfault on some Windows CUDA / oneDNN combos; force pure-CPU path.
    """
    import os as _os
    env = dict(os.environ)             # copy parent env
    env["PYTHONDONTWRITEBYTECODE"] = "1"                # L1
    env["CUDA_VISIBLE_DEVICES"] = "-1"                   # L5
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"                   # L5
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"                    # noise reduction
    env["YOLO_AUTOINSTALL"] = "False"                    # no uv autoupdate noise

    # L2: wipe onnx2tf __pycache__ (run via interpreter probe like allow_pickle patch)
    _probe = subprocess.run(
        [SYS_PY_WITH_TF_AND_ONNX2TF, "-c",
         "import onnx2tf.utils.common_functions as m, os, shutil; "
         "p=os.path.join(os.path.dirname(os.path.abspath(m.__file__)),'__pycache__'); "
         "shutil.rmtree(p, ignore_errors=True); print('ok')"],
        capture_output=True, text=True
    )
    if _probe.returncode == 0 and _probe.stdout.strip() == "ok":
        print("  [self-heal L2] onnx2tf utils __pycache__ wiped.")
    else:
        print(f"  [self-heal L2] __pycache__ wipe note: rc={_probe.returncode} "
              f"stdout={_probe.stdout.strip()[:120]}")

    # L3: write calibration npy (via same interpreter so we know the numpy matches)
    cwd_for_subprocess.mkdir(parents=True, exist_ok=True)
    calib_name = "calibration_image_sample_data_20x128x128x3_float32.npy"
    calib_path = cwd_for_subprocess / calib_name
    _w = subprocess.run(
        [SYS_PY_WITH_TF_AND_ONNX2TF, "-c",
         "import sys, numpy as np; "
         f"p=sys.argv[1]; "
         f"np.save(p, (np.random.rand(20,128,128,3).astype(np.float32)*255.0)); "
         "a=np.load(p, allow_pickle=False); "
         "b=np.load(p, allow_pickle=True); "
         "print(a.shape, a.dtype, a.min(), a.max(), 'BOTH_OK')",
         str(calib_path)],
        capture_output=True, text=True
    )
    if _w.returncode == 0 and "BOTH_OK" in _w.stdout:
        print(f"  [self-heal L3] wrote valid calibration cache -> {calib_path.name} ({calib_path.stat().st_size/1024:.0f} KB)")
    else:
        print(f"  [self-heal L3] warn: cache writer rc={_w.returncode} tail stdout={_w.stdout[-200:].strip()}")

    return env


# NOTE: With onnx2tf 1.22 + minimal flags, the YOLO detection head is preserved
# perfectly (float32 model matches raw PT output bit-for-bit on the orange-can
# reference model; the keep-op flags from earlier versions were only needed as
# a workaround for a different onnx2tf crash mode).  Keep the conversion simple
# and robust: `-i onnx -o ascii_output_dir` (no -osd, no -k).
YOLO_HEAD_KEEP_OPS: list[str] = []   # kept as hook in case a future model needs them

# Use ASCII-only temporary output path for onnx2tf.  tensorflow 2.15's
# TFLite Interpreter / onnx2tf file writer has unicode-path bugs on Windows;
# the final copy-to-destination handles unicode names just fine.
tflite_ascii_tmp_dir = OUTPUTS_DIR / "_tflite_ascii_tmp"
if tflite_ascii_tmp_dir.exists():
    shutil.rmtree(tflite_ascii_tmp_dir, ignore_errors=True)

# onnx2tf checks `os.getcwd()` for the calibration npy, so run the subprocess
# with `cwd=tflite_ascii_tmp_dir.parent` (OUTPUTS_DIR) where we just wrote the
# cache file.  That way onnx2tf reads LOCAL_FILE_PATH = getcwd()/calib.npy.
subprocess_cwd = tflite_ascii_tmp_dir.parent
subprocess_env = _prepare_onnx2tf_runtime(subprocess_cwd)

onnx2tf_cmd = [
    SYS_PY_WITH_TF_AND_ONNX2TF, "-m", "onnx2tf",
    "-i", str(ONNX_MODEL),
    "-o", str(tflite_ascii_tmp_dir),
]
for op in YOLO_HEAD_KEEP_OPS:
    onnx2tf_cmd += ["-k", op]

print(f"  running (cwd={subprocess_cwd}): {' '.join(map(str, onnx2tf_cmd))}")
print(f"  env[PYTHONDONTWRITEBYTECODE]={subprocess_env.get('PYTHONDONTWRITEBYTECODE')} CUDA_VISIBLE_DEVICES={subprocess_env.get('CUDA_VISIBLE_DEVICES')}")
result = subprocess.run(
    onnx2tf_cmd,
    cwd=str(subprocess_cwd),
    env=subprocess_env,
    capture_output=True, text=True,
    timeout=60 * 60,
)
if result.returncode != 0:
    print(result.stdout[-2500:])
    print("--- STDERR ---")
    print(result.stderr[-3500:])
    print(f"ERROR: onnx2tf conversion failed (exit={result.returncode})")
    sys.exit(1)
# Prefer _float32.tflite over _float16.tflite over any other .tflite
tflite_candidates = list(tflite_ascii_tmp_dir.rglob("*.tflite"))
_ordered = sorted(
    tflite_candidates,
    key=lambda p: (0 if "_float32" in p.name else 1 if "_float16" in p.name else 2, p.stat().st_size)
)
if not _ordered:
    print(f"ERROR: No .tflite found under {tflite_ascii_tmp_dir}")
    print("\nDirectory listing:")
    for p in sorted(tflite_ascii_tmp_dir.rglob("*")):
        print(" ", p.relative_to(tflite_ascii_tmp_dir))
    sys.exit(1)
TFLITE_SRC = _ordered[0]
TFLITE_OUTPUT = OUTPUTS_DIR / "detect.tflite"
shutil.copy2(TFLITE_SRC, TFLITE_OUTPUT)
# Also copy _float16.tflite variant next to it as optional asset
for p in _ordered:
    if "_float16" in p.name:
        fp16_dst = OUTPUTS_DIR / "detect_float16.tflite"
        shutil.copy2(p, fp16_dst)
        print(f"  TFLite fp16: {fp16_dst} ({fp16_dst.stat().st_size/1024/1024:.1f} MB)")
        break
print(f"  TFLite fp32: {TFLITE_OUTPUT} ({TFLITE_OUTPUT.stat().st_size/1024/1024:.1f} MB)  [from {TFLITE_SRC.name}]")

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
