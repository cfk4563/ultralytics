from ultralytics import YOLO


if __name__ == '__main__':

    model_path = '/mnt/workspace/yolo/ultralytics/runs/detect/train24/weights/best.pt'
    img_path = "/mnt/workspace/yolo/ultralytics/datasets/LROC/images/test/cropped_image_2259.jpg"

    model = YOLO(model_path)
    results = model.predict(source=img_path, save=True)