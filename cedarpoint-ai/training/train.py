from ultralytics import YOLO


def main():
    print("Loading YOLOv8s model...")

    model = YOLO("yolov8s.pt")

    print("Starting training on Jetson Orin Nano...")

    results = model.train(
        data="coco.yaml",
        epochs=30,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=2,
        cache=False,
        project="runs",
        name="yolov8s_orin",
        pretrained=True,
        patience=20
    )

    print("Training complete!")
    print(results)


if __name__ == "__main__":
    main()