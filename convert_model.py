#!/usr/bin/env python3
"""
convert_model.py — PlantDiag Webapp version 1.2.2
=================================================

Converts your Colab-trained plant_disease_model.tflite into the
TensorFlow.js graph format the web app can load in a browser:

    plant_disease_model.tflite   ──►  model/model.json
                                       model/group1-shard1ofN.bin ...
                                       model/labels.txt

The browser CANNOT execute .tflite files — this one-time conversion is
REQUIRED before using index.html.

HOW TO RUN (from this folder):

    python convert_model.py

Optional arguments:

    python convert_model.py --tflite plant_disease_model.tflite --labels labels.txt --out model

Requirements:
    Python 3.9 – 3.12
    pip install tensorflow tensorflowjs     (the script installs tensorflowjs
                                             automatically if it's missing)

After conversion, serve the app over HTTP (see README.md), e.g.:

    python -m http.server 8000
    # then open  http://localhost:8000
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Defaults — files sit NEXT TO this script
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TFLITE = SCRIPT_DIR / "plant_disease_model.tflite"
DEFAULT_LABELS = SCRIPT_DIR / "labels.txt"
DEFAULT_OUT_DIR = SCRIPT_DIR / "model"


def banner(text: str) -> None:
    line = "=" * len(text)
    print(f"\n{line}\n{text}\n{line}")


def ensure_tensorflowjs() -> None:
    """Install tensorflowjs (and tensorflow if needed) into the current env."""
    if importlib.util.find_spec("tensorflowjs") is None:
        print("[1/3] tensorflowjs not found — installing (this can take a minute)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "tensorflowjs"]
        )
    else:
        print("[1/3] tensorflowjs already installed.")

    if importlib.util.find_spec("tensorflow") is None:
        print("      tensorflow not found — installing it too...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "tensorflow"]
        )


def convert_tflite(tflite_path: Path, out_dir: Path) -> None:
    """Run tensorflowjs_converter on the .tflite file."""
    print(f"[2/3] Converting {tflite_path.name} → {out_dir}/ ...")
    cmd = [
        sys.executable,
        "-m",
        "tensorflowjs.converters.converter",
        "--input_format=tflite",
        str(tflite_path),
        str(out_dir),
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print("\n[ERROR] Conversion failed.")
        print(
            "Common causes:\n"
            "  • Fully INT8-quantized model with ops TF.js cannot map.\n"
            "    Fix: re-export a float32 TFLite in Colab:\n"
            "        converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')\n"
            "        tflite_model = converter.convert()  # float32, no post-training quant\n"
            "  • Model uses custom ops — convert with Select TFOps:\n"
            "        tensorflowjs_converter \\\n"
            "            --input_format=tflite \\\n"
            "            --allow_custom_ops \\\n"
            "            model.tflite model/\n"
        )
        raise SystemExit(exc.returncode) from exc


def copy_labels(labels_path: Path, out_dir: Path) -> None:
    """Copy labels.txt into the model folder (fetched by app.js at runtime)."""
    dest = out_dir / "labels.txt"
    if labels_path.exists():
        shutil.copy2(labels_path, dest)
        count = sum(
            1
            for line in dest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        print(f"[3/3] Copied {labels_path.name} → {dest}  ({count} labels)")
    else:
        placeholder = (
            "# labels.txt not found next to convert_model.py.\n"
            "# Paste your Colab-exported labels here, ONE PER LINE, in model output\n"
            "# order (class index 0 first), then refresh the web app.\n"
        )
        dest.write_text(placeholder, encoding="utf-8")
        print(
            f"[3/3] WARNING: {labels_path} not found.\n"
            f"      Wrote a placeholder to {dest}.\n"
            "      Copy your Colab labels.txt over it (one label per line, model order)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert plant_disease_model.tflite to TensorFlow.js format for the PlantDiag web app."
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=DEFAULT_TFLITE,
        help=f"Path to the .tflite model (default: {DEFAULT_TFLITE.name})",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help=f"Path to labels.txt (default: {DEFAULT_LABELS.name})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output folder for the TF.js model (default: model/)",
    )
    args = parser.parse_args()

    banner("PlantDiag — .tflite → TensorFlow.js converter")

    tflite_path: Path = args.tflite.expanduser().resolve()
    labels_path: Path = args.labels.expanduser().resolve()
    out_dir: Path = args.out.expanduser().resolve()

    if not tflite_path.exists():
        print(f"[ERROR] Model file not found: {tflite_path}")
        print(
            "Place your Colab-exported plant_disease_model.tflite NEXT TO this script,\n"
            "or pass its path:  python convert_model.py --tflite C:\\path\\to\\model.tflite"
        )
        raise SystemExit(1)

    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    print(f"      Model : {tflite_path}  ({size_mb:.1f} MB)")
    print(f"      Labels: {labels_path}")
    print(f"      Output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_tensorflowjs()
    convert_tflite(tflite_path, out_dir)
    copy_labels(labels_path, out_dir)

    produced = sorted(p for p in out_dir.iterdir() if p.is_file())
    total_mb = sum(p.stat().st_size for p in produced) / (1024 * 1024)
    banner("DONE — model/ folder is ready")
    for p in produced:
        print(f"  • {p.relative_to(SCRIPT_DIR)}  ({p.stat().st_size / 1024:.0f} KB)")
    print(f"\nTotal: {total_mb:.1f} MB in {out_dir}")
    print(
        "\nNext steps:\n"
        "  1. python -m http.server 8000      (or use VS Code Live Server)\n"
        "  2. Open http://localhost:8000 in your browser\n"
        "  3. Upload a leaf photo → Analyze\n"
        "\nNOTE: do NOT open index.html directly with file:// — browsers block\n"
        "      fetching model weights from the filesystem. Always serve over HTTP.\n"
    )


if __name__ == "__main__":
    main()
