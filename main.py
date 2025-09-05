#!/usr/bin/env python3
"""
main.py – Unified entrypoint for PALM-LAY training, evaluation, and conversion.

Usage
-----
# Train YOLOv8
python main.py train-yolo --model yolov8s.pt --epochs 100 --batch 8

# Train YOLOv9
python main.py train-yolo --model yolov9s.pt

# Train YOLOv11
python main.py train-yolo --model yolov11s.pt

# Train YOLOv10
python main.py train-yolo10 --weights yolov10n.pt

# Train DETR
python main.py train-detr --epochs 50 --batch 4

# Evaluate COCO (for DETR/RF-DETR)
python main.py eval-coco --gt coco/val.json --pred runs_detr/preds.json

# Convert YOLO -> COCO
python main.py yolo2coco \
    --images data/images/train \
    --labels data/labels/train \
    --yaml configs/palm_lay_yolo.yaml \
    --out coco/train.json

# Convert COCO -> YOLO
python main.py coco2yolo \
    --coco coco/val.json \
    --images data/images/val \
    --yaml configs/palm_lay_yolo.yaml \
    --out data/labels/val

# Launch RF-DETR (external repo required)
python main.py train-rfdetr
"""

import argparse
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

def run_script(script, args):
    cmd = [sys.executable, str(SCRIPTS / script)] + args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="PALM-LAY unified entrypoint")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # YOLOv8/9/11
    sub.add_parser("train-yolo", help="Train YOLOv8/9/11 with Ultralytics")

    # YOLOv10
    sub.add_parser("train-yolo10", help="Train YOLOv10")

    # DETR
    sub.add_parser("train-detr", help="Train DETR (HuggingFace)")

    # COCO eval
    sub.add_parser("eval-coco", help="Evaluate predictions vs COCO ground truth")

    # YOLO ↔ COCO
    sub.add_parser("yolo2coco", help="Convert YOLO TXT to COCO JSON")
    sub.add_parser("coco2yolo", help="Convert COCO JSON to YOLO TXT")

    # RF-DETR
    sub.add_parser("train-rfdetr", help="Launch RF-DETR training (via shell script)")

    args, rest = parser.parse_known_args()

    if args.cmd == "train-yolo":
        run_script("train_yolo.py", rest)
    elif args.cmd == "train-yolo10":
        run_script("train_yolo10.py", rest)
    elif args.cmd == "train-detr":
        run_script("train_detr.py", rest)
    elif args.cmd == "eval-coco":
        run_script("eval_coco.py", rest)
    elif args.cmd in ("yolo2coco", "coco2yolo"):
        run_script("yolo_coco_convert.py", [args.cmd] + rest)
    elif args.cmd == "train-rfdetr":
        # RF-DETR is a shell launcher
        cmd = ["bash", str(SCRIPTS / "train_rfdetr.sh")] + rest
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
