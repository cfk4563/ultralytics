from ultralytics import YOLO

# Load a model
# # model = YOLO("yolov8n.yaml")  # build a new model from scratch
# if __name__ == '__main__':
#     model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
if __name__ == '__main__':

    model = YOLO('LSKA.yaml')  # build a new model from YAML
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model = YOLO('LSKA.yaml').load('yolov8n.pt')

    results = model.train(data="mdcd.yaml",epochs=150,imgsz=640)
    #
    # model =  YOLO('../runs/detect/train4/weights/best.pt')
    # model.train()