# scripts/train_yolo10.py
import argparse
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="configs/palm_lay_yolo.yaml")
    ap.add_argument("--weights", type=str, default="yolov10n.pt", help="yolov10n/s/m/l/x.pt")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", type=str, default="runs_yolov10")
    ap.add_argument("--name", type=str, default="exp")
    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "yolov10", "train",
        f"data={args.data}",
        f"model={args.weights}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"project={args.project}",
        f"name={args.name}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
