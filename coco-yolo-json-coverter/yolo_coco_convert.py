#!/usr/bin/env python3
"""
yolo_coco_convert.py
Bidirectional converter between YOLO (TXT) and COCO (JSON) formats.

Usage
-----
# YOLO -> COCO
python scripts/yolo_coco_convert.py yolo2coco \
  --images data/images/train \
  --labels data/labels/train \
  --yaml   configs/palm_lay_yolo.yaml \
  --out    coco/train.json

python scripts/yolo_coco_convert.py yolo2coco \
  --images data/images/val \
  --labels data/labels/val \
  --yaml   configs/palm_lay_yolo.yaml \
  --out    coco/val.json

# COCO -> YOLO
python scripts/yolo_coco_convert.py coco2yolo \
  --coco   coco/val.json \
  --images data/images/val \
  --yaml   configs/palm_lay_yolo.yaml \
  --out    data/labels/val

Notes
-----
- YOLO labels: one .txt per image with lines:
    <class> <cx> <cy> <w> <h>  (normalized to [0,1])
- COCO boxes: [x, y, w, h] in absolute pixels (top-left origin).
- The YAML file must define 'names:' in index order (0..N-1).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import yaml
from collections import defaultdict


# ---------- Utilities ----------

def read_classes_from_yaml(yaml_path: str) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # supports both dict {0: name, ...} and list [name0, ...]
    names = cfg.get("names")
    if isinstance(names, dict):
        # ensure sorted by integer key
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if not isinstance(names, list):
        raise ValueError("Could not parse 'names' from YAML. Expect list or {idx:name} dict.")
    return names


def image_size(img_path: Path) -> Tuple[int, int]:
    with Image.open(img_path) as im:
        w, h = im.size
    return w, h


def yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h) -> List[float]:
    x = (cx - w / 2.0) * img_w
    y = (cy - h / 2.0) * img_h
    return [float(x), float(y), float(w * img_w), float(h * img_h)]


def coco_to_yolo_bbox(x, y, w, h, img_w, img_h) -> List[float]:
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    ww = w / img_w
    hh = h / img_h
    return [float(cx), float(cy), float(ww), float(hh)]


def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])


# ---------- YOLO -> COCO ----------

def yolo2coco(images_dir: str, labels_dir: str, yaml_path: str, out_json: str) -> None:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    class_names = read_classes_from_yaml(yaml_path)
    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]

    images = []
    annotations = []
    ann_id = 1  # COCO annotation id starts at 1 (convention)

    img_paths = list_images(images_dir)
    if not img_paths:
        raise RuntimeError(f"No images found in: {images_dir}")

    # map basenames to label paths
    for idx, img_path in enumerate(img_paths, start=1):
        w, h = image_size(img_path)
        rel_name = img_path.relative_to(images_dir).as_posix()  # keep subfolders if any

        img_info = {
            "id": idx,
            "file_name": rel_name,
            "width": w,
            "height": h,
        }
        images.append(img_info)

        # find matching label file
        lbl_path = labels_dir / Path(rel_name).with_suffix(".txt")
        if not lbl_path.exists():
            # image with zero annotations is still valid; skip annotations
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Malformed YOLO line in {lbl_path}: '{line}'")
                c, cx, cy, ww, hh = parts
                c = int(c)
                cx, cy, ww, hh = map(float, (cx, cy, ww, hh))
                if c < 0 or c >= len(class_names):
                    raise ValueError(f"Class id {c} out of range in {lbl_path}")

                x, y, W, H = yolo_to_coco_bbox(cx, cy, ww, hh, w, h)
                ann = {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": c,
                    "bbox": [x, y, W, H],
                    "area": float(W * H),
                    "iscrowd": 0,
                }
                annotations.append(ann)
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    print(f"[YOLO→COCO] Wrote {out_json} with {len(images)} images and {len(annotations)} annotations.")


# ---------- COCO -> YOLO ----------

def coco2yolo(coco_json: str, images_dir: str, yaml_path: str, out_labels_dir: str) -> None:
    images_dir = Path(images_dir)
    out_labels_dir = Path(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_classes_from_yaml(yaml_path)
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Validate categories align with YAML names (by name, order is preserved by YAML)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    # Build mapping: category_id -> YOLO class index
    # If names match exactly, use index of that name in YAML names
    cat_id_to_yolo = {}
    for cid, cname in cat_id_to_name.items():
        if cname not in class_names:
            raise ValueError(f"COCO category '{cname}' not found in YAML names: {class_names}")
        cat_id_to_yolo[cid] = class_names.index(cname)

    # Index images and annotations
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    anns_by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    # Write one YOLO file per image
    count_files = 0
    count_anns = 0
    for img_id, im in images_by_id.items():
        file_name = im["file_name"]
        img_path = images_dir / file_name
        if not img_path.exists():
            # Try fallback by basename search if structure differs
            fallback = images_dir / Path(file_name).name
            if fallback.exists():
                img_path = fallback
            else:
                print(f"[WARN] Image not found: {img_path} (skipping)")
                continue

        W, H = im.get("width"), im.get("height")
        if not W or not H:
            # if missing, compute from actual image
            W, H = image_size(img_path)

        # Prepare label path (mirror subfolders)
        lbl_path = out_labels_dir / Path(file_name).with_suffix(".txt")
        lbl_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for a in anns_by_img.get(img_id, []):
            cid = a["category_id"]
            if cid not in cat_id_to_yolo:
                # Skip unknown categories
                continue
            yolo_c = cat_id_to_yolo[cid]
            x, y, w, h = a["bbox"]
            cx, cy, ww, hh = coco_to_yolo_bbox(x, y, w, h, W, H)
            # clamp to [0,1] in case of rounding
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            ww = min(max(ww, 0.0), 1.0)
            hh = min(max(hh, 0.0), 1.0)
            lines.append(f"{yolo_c} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
            count_anns += 1

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        count_files += 1

    print(f"[COCO→YOLO] Wrote {count_files} label files to {out_labels_dir} with {count_anns} boxes.")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO <-> COCO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("yolo2coco", help="Convert YOLO TXT labels to COCO JSON")
    s1.add_argument("--images", required=True, help="Path to images root (train/ or val/)")
    s1.add_argument("--labels", required=True, help="Path to YOLO labels root (train/ or val/)")
    s1.add_argument("--yaml",   required=True, help="YOLO dataset YAML defining 'names'")
    s1.add_argument("--out",    required=True, help="Output COCO JSON path")

    s2 = sub.add_parser("coco2yolo", help="Convert COCO JSON to YOLO TXT labels")
    s2.add_argument("--coco",   required=True, help="COCO JSON path")
    s2.add_argument("--images", required=True, help="Path to images root (train/ or val/)")
    s2.add_argument("--yaml",   required=True, help="YOLO dataset YAML defining 'names'")
    s2.add_argument("--out",    required=True, help="Output labels folder (will be created)")

    args = parser.parse_args()

    if args.cmd == "yolo2coco":
        yolo2coco(args.images, args.labels, args.yaml, args.out)
    elif args.cmd == "coco2yolo":
        coco2yolo(args.coco, args.images, args.yaml, args.out)


if __name__ == "__main__":
    main()
