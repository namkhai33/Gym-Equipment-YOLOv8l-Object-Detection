import os
import sys
from multiprocessing import freeze_support
import torch
import warnings
warnings.filterwarnings('ignore')

def main() -> None:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        sys.exit(1)

    model = YOLO('yolov8l.pt')
    try:
        model.train(
            data='data.yaml',
            epochs=100,
            batch=16,
            imgsz=640,
            device='cuda',
            patience=50,
            cache=True,
            cos_lr=True,
            name='train',
            workers=4,
            save_period=1,
            amp=True,
            plots=True,
            lr0=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )
    except Exception as e:
        print(f"Training error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    freeze_support()
    main()