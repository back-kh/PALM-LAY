# scripts/eval_coco.py
import argparse, json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="coco/val.json")
    ap.add_argument("--pred", required=True, help="predictions.json (COCO format)")
    args = ap.parse_args()

    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(args.pred)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

if __name__ == "__main__":
    main()
