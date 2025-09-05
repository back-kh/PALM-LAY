#!/usr/bin/env bash
set -e

# Example template — replace URL/branch/config with the official RF-DETR repo you use.
REPO_URL=${REPO_URL:-"https://github.com/your-org/rf-detr.git"}
REPO_DIR=${REPO_DIR:-"external/rf-detr"}
CONFIG=${CONFIG:-"configs/rf_detr_r50_8xb2_800_50e_palm-lay.py"}

python -V
mkdir -p external
[ -d "$REPO_DIR" ] || git clone --depth 1 "$REPO_URL" "$REPO_DIR"

cd "$REPO_DIR"
pip install -r requirements.txt

# Typical training entrypoint; many repos use "tools/train.py"
python tools/train.py "$CONFIG" \
  --work-dir "../../runs_rfdetr" \
  --cfg-options \
  data.train.ann_file="../../coco/train.json" \
  data.train.img_prefix="../../data/images/train" \
  data.val.ann_file="../../coco/val.json" \
  data.val.img_prefix="../../data/images/val"

# Example evaluation (edit to match repo)
# python tools/test.py "$CONFIG" path/to/weights.pth --eval bbox
