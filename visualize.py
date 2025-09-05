#!/usr/bin/env python3
"""
scripts/visualize.py
Visualize YOLO or COCO annotations/predictions on images.

Usage
-----
# 1) Visualize YOLO ground truth labels
python scripts/visualize.py \
  --images data/images/val \
  --yolo data/labels/val \
  --yaml  configs/palm_lay_yolo.yaml \
  --out   viz/yolo_gt

# 2) Visualize YOLO predictions (same TXT format)
python scripts/visualize.py \
  --images data/images/val \
  --yolo runs_yolo/exp/labels \
  --yaml configs/palm_lay_yolo.yaml \
  --out  viz/yolo_pred \
  --pred  # adds score parsing if you put 6th value = score (optional)

# 3) Visualize COCO ground truth
python scripts/visualize.py \
  --images data/images/val \
  --coco   coco/val.json \
  --yaml   configs/palm_lay_yolo.yaml \
  --out    viz/coco_gt

# 4) Visualize COCO predictions
python scripts/visualize.py \
  --images data/images/val \
  --coco   runs_detr/preds.json \
  --yaml   configs/palm_lay_yolo.yaml \
  --out    viz/coco_pred \
  --pred

Notes
-----
- YOLO labels: class cx cy w h (normalized). If you add a 6th value, it is treated as a score when --pred is set.
- COCO predictions: standard list of dicts with keys: image_id, category_id, bbox=[x,y,w,h], score.
- COCO ground truth: standard dict with "images","annotations","categories".
- Output images are saved with overlaid boxes + labels.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import random

import cv2
import yaml
from PIL import Image

# ---------------------------
# Utilities
# ---------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def load_names(yaml_path: str) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        # ensure index order
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if not isinstance(names, list):
        raise ValueError("Could not parse 'names' from YAML (need list or {idx:name}).")
    return names

def list_images(images_root: Path) -> List[Path]:
    return sorted([p for p in images_root.rglob("*") if p.suffix.lower() in IMG_EXTS])

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_image_cv2(path: Path):
    img = cv2.imdecode(np_from_file(path), cv2.IMREAD_COLOR)
    if img is None:
        # Fallback: cv2.imread
        img = cv2.imread(str(path))
    return img

def np_from_file(path: Path):
    import numpy as np
    data = np.fromfile(str(path), dtype=np.uint8)
    return data

def get_img_size_pil(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)

def make_palette(n: int) -> List[Tuple[int, int, int]]:
    random.seed(1234)
    return [(random.randint(30, 230), random.randint(30, 230), random.randint(30, 230)) for _ in range(n)]

def draw_box(
    img,
    x1, y1, x2, y2,
    color=(0,255,0),
    label:str=None,
    score:float=None,
    thickness:int=2,
    font_scale:float=0.5
):
    # clamp
    H, W = img.shape[:2]
    x1 = max(0, min(int(x1), W-1))
    y1 = max(0, min(int(y1), H-1))
    x2 = max(0, min(int(x2), W-1))
    y2 = max(0, min(int(y2), H-1))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label is not None:
        text = label
        if score is not None:
            text = f"{label} {score:.2f}"
        # Text background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        th = th + baseline + 4
        cv2.rectangle(img, (x1, y1 - th if y1 - th > 0 else y1 + th), (x1 + tw + 4, y1), color, -1)
        # Text
        ty = y1 - 4 if y1 - th > 0 else y1 + th - 4
        cv2.putText(img, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1, cv2.LINE_AA)

# ---------------------------
# YOLO helpers
# ---------------------------

def yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    return x1, y1, x2, y2

def read_yolo_txt(txt_path: Path) -> List[List[float]]:
    """
    Returns list of entries. Supports:
    - class cx cy w h
    - class cx cy w h score  (if predictions)
    """
    entries = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) not in (5, 6):
                # Skip malformed lines
                continue
            vals = [float(x) for x in parts]
            entries.append(vals)
    return entries

# ---------------------------
# COCO helpers
# ---------------------------

def load_coco(coco_path: Path):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco

def index_coco_gt(coco: dict):
    images = {im["id"]: im for im in coco.get("images", [])}
    anns_by_img = {}
    for a in coco.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    return images, anns_by_img, cat_id_to_name

def index_coco_pred(pred_list: List[dict]):
    by_img = {}
    for a in pred_list:
        by_img.setdefault(a["image_id"], []).append(a)
    return by_img

# ---------------------------
# Main visualization
# ---------------------------

def vis_yolo(images_root: Path, labels_root: Path, class_names: List[str], out_root: Path, is_pred: bool):
    ensure_dir(out_root)
    palette = make_palette(len(class_names))
    img_paths = list_images(images_root)
    import numpy as np

    for img_path in img_paths:
        rel = img_path.relative_to(images_root)
        out_path = out_root / rel
        ensure_dir(out_path.parent)

        img = cv2.imdecode(np_from_file(img_path), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot read image: {img_path}")
            continue

        W, H = img.shape[1], img.shape[0]
        txt_path = labels_root / rel.with_suffix(".txt")
        boxes = []
        if txt_path.exists():
            rows = read_yolo_txt(txt_path)
            for r in rows:
                if len(r) == 5:
                    c, cx, cy, ww, hh = r
                    score = None
                else:
                    c, cx, cy, ww, hh, score = r
                c = int(c)
                if c < 0 or c >= len(class_names):
                    continue
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, ww, hh, W, H)
                boxes.append((x1, y1, x2, y2, c, score))
        else:
            # no labels -> just copy image
            pass

        for (x1, y1, x2, y2, c, score) in boxes:
            draw_box(img, x1, y1, x2, y2, color=palette[c], label=class_names[c], score=score if is_pred else None)

        # write
        out_bytes = cv2.imencode(".jpg", img)[1]
        out_path = out_path.with_suffix(".jpg")
        out_path.write_bytes(out_bytes.tobytes())

def vis_coco(images_root: Path, coco_path: Path, class_names: List[str], out_root: Path, is_pred: bool):
    ensure_dir(out_root)
    palette = make_palette(len(class_names))
    import numpy as np

    coco = load_coco(coco_path)
    # If it's predictions (list), convert to dict
    if isinstance(coco, list):
        # predictions list
        by_img = index_coco_pred(coco)
        # We need images mapping. We'll try to find them by file name in folder.
        # If your predictions include "file_name", you can adapt here.
        # Otherwise, require a paired GT coco for proper mapping.
        # Fallback: infer image_id by enumerating images in the folder in sorted order (not ideal).
        print("[INFO] COCO predictions provided as a list — expecting 'image_id' integers matching ground-truth IDs.")
        # Since we don't have GT 'images', we’ll fall back to file-name mapping by index order:
        # WARNING: This requires a consistent image_id numbering that matches GT (preferred).
        # If not available, consider passing GT json instead, or postprocess your preds to include file_name.
        # Here we skip the file-name mapping and assume users provide GT json for image/file mapping.
        raise RuntimeError("For prediction visualization with COCO list, please pass GT COCO to resolve image_id->file_name.")
    else:
        # ground-truth style dict
        images, anns_by_img, cat_id_to_name = index_coco_gt(coco)
        # Build cat_id -> yolo index by name (in case names must align to README order)
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        cat_id_to_color_idx = {}
        for cid, cname in cat_id_to_name.items():
            cat_id_to_color_idx[cid] = name_to_idx.get(cname, 0)

        # process each image
        for img_id, im in images.items():
            file_name = im["file_name"]
            img_path = images_root / file_name
            if not img_path.exists():
                # try basename fallback
                fb = images_root / Path(file_name).name
                if fb.exists():
                    img_path = fb
                else:
                    print(f"[WARN] image file not found for COCO image_id={img_id}: {img_path}")
                    continue

            img = cv2.imdecode(np_from_file(img_path), cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] cannot read {img_path}")
                continue

            anns = anns_by_img.get(img_id, [])
            for a in anns:
                cid = a["category_id"]
                x, y, w, h = a["bbox"]
                x1, y1, x2, y2 = x, y, x + w, y + h
                cname = cat_id_to_name.get(cid, str(cid))
                cidx = cat_id_to_color_idx.get(cid, 0)
                draw_box(img, x1, y1, x2, y2, color=palette[cidx], label=cname, score=(a.get("score") if is_pred else None))

            out_path = (Path(out_root) / file_name).with_suffix(".jpg")
            ensure_dir(out_path.parent)
            out_bytes = cv2.imencode(".jpg", img)[1]
            out_path.write_bytes(out_bytes.tobytes())

def main():
    ap = argparse.ArgumentParser(description="Visualize YOLO/COCO annotations on images.")
    ap.add_argument("--images", required=True, help="Images root folder")
    ap.add_argument("--yolo",   default=None, help="YOLO labels root (TXT files)")
    ap.add_argument("--coco",   default=None, help="COCO JSON (GT dict or predictions list)")
    ap.add_argument("--yaml",   required=True, help="YAML with 'names' list/dict")
    ap.add_argument("--out",    required=True, help="Output folder for visualizations")
    ap.add_argument("--pred",   action="store_true", help="Treat inputs as predictions (shows scores if available)")
    args = ap.parse_args()

    images_root = Path(args.images)
    out_root = Path(args.out)
    class_names = load_names(args.yaml)

    if args.yolo and args.coco:
        raise ValueError("Please specify either --yolo or --coco, not both.")

    if args.yolo:
        vis_yolo(images_root, Path(args.yolo), class_names, out_root, is_pred=args.pred)
    elif args.coco:
        vis_coco(images_root, Path(args.coco), class_names, out_root, is_pred=args.pred)
    else:
        raise ValueError("You must specify one of --yolo or --coco.")

if __name__ == "__main__":
    # Lazy import numpy to keep top-level import minimal
    import numpy as np  # noqa: E402
    main()
