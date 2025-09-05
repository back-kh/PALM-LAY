# scripts/train_yolo.py
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="configs/palm_lay_yolo.yaml")
    ap.add_argument("--model", type=str, default="yolov8s.pt",
                    help="yolov8*.pt / yolov9*.pt / yolov11*.pt (Ultralytics)")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", type=str, default="runs_yolo")
    ap.add_argument("--name", type=str, default="exp")
    args = ap.parse_args()

    model = YOLO(args.model)  # auto-downloads weights if needed
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        verbose=True
    )

    # Evaluate on val set
    model.val(data=args.data, imgsz=args.imgsz, split="val")

if __name__ == "__main__":
    main()
