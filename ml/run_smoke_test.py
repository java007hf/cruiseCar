"""Standalone smoke-test runner — reuse the same run_post_train_smoke_test from train_server."""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from train_server import run_post_train_smoke_test, OUTPUTS_DIR  # noqa: E402


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pt", type=Path, default=None,
                   help="Path to PT model (default: newest *.pt in outputs/)")
    p.add_argument("--tflite", type=Path, default=OUTPUTS_DIR / "detect.tflite")
    p.add_argument("--yaml", type=Path, default=None,
                   help="Path to dataset.yaml (default: try to match pt name or newest dataset yaml)")
    p.add_argument("--conf", type=float, default=0.20)
    args = p.parse_args()

    # auto-pick PT model: newest stemmed_*.pt
    if args.pt is None or not args.pt.exists():
        candidates = sorted(
            [p for p in OUTPUTS_DIR.glob("*_*.pt")],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not candidates:
            print("ERROR: no *_*.pt found in", OUTPUTS_DIR)
            sys.exit(1)
        args.pt = candidates[0]
        print(f"[auto] Using latest PT model: {args.pt}")

    # auto-pick yaml
    if args.yaml is None or not args.yaml.exists():
        from train_server import DATASETS_DIR
        stem = args.pt.stem  # "一瓶可乐_20260805_194757"
        run_id = stem.split("_")[-1] if "_" in stem else None
        if run_id:
            cand = DATASETS_DIR / run_id / "dataset.yaml"
            if cand.exists():
                args.yaml = cand
                print(f"[auto] Using dataset.yaml: {args.yaml}")
        if args.yaml is None or not args.yaml.exists():
            # fallback: newest dataset.yaml under datasets
            ys = sorted(DATASETS_DIR.glob("*/dataset.yaml"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
            if ys:
                args.yaml = ys[0]
                print(f"[auto] Fallback dataset.yaml: {args.yaml}")

    print()
    print("Running run_post_train_smoke_test:")
    print(f"  pt      = {args.pt} (exists={args.pt.exists()})")
    print(f"  tflite  = {args.tflite} (exists={args.tflite.exists() if args.tflite else 'N/A'})")
    print(f"  yaml    = {args.yaml} (exists={args.yaml.exists() if args.yaml else 'N/A'})")
    print(f"  conf    = {args.conf}")
    print()
    res = run_post_train_smoke_test(args.pt, args.tflite, args.yaml, conf_thresh=args.conf)
    print()
    print("RESULT SUMMARY:")
    print(f"  has_images: {res.get('has_images')}, pt_ok={res.get('pt_ok')}, tflite_ok={res.get('tflite_ok')}")
    if res.get("report_path"):
        print(f"  report:     {res['report_path']}")
    if res.get("output_dir"):
        print(f"  output_dir: {res['output_dir']}")
    if res.get("total_images"):
        print(f"  pt hits:    {res.get('pt_hits', 0)}/{res['total_images']}")
        print(f"  tflite hits:{res.get('tflite_hits', 0)}/{res['total_images']}")
    for per in res.get("per_image", []):
        name = per["name"]
        pt = per.get("pt")
        tf = per.get("tflite")
        def _fmt(x):
            if not isinstance(x, dict):
                return f"{x!r}"
            if "error" in x:
                return f"ERR: {x['error'][:80]}"
            return f"count={x.get('count')}, top_conf={x.get('top_conf', 0):.4f}, confs={x.get('confs', [])[:3]}"
        print(f"  - {name}: PT={{{_fmt(pt)}}}  TFLite={{{_fmt(tf)}}}")
        if per.get("compare_path"):
            print(f"      compare: {per['compare_path']}")


if __name__ == "__main__":
    main()
