from ultralytics import YOLO

# Load a model
# # model = YOLO("yolov8n.yaml")  # build a new model from scratch
# if __name__ == '__main__':
#     model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
if __name__ == '__main__':

    model = YOLO('../runs/detect/train11/weights/best.pt')

    results = model.predict(source="C:/Users/Administrator/Desktop/yolo/ultralytics/datasets/MDCD/234.jpg",save=True)