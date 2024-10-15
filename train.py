from ultralytics import YOLO


if __name__ == '__main__':
    # for i in range(3):
    #     print(f"第{i+1}次训练")
    model = YOLO('yolo11n.yaml').load("yolo11n.pt")
    results = model.train(data="LROC.yaml",epochs=200,imgsz=640)

