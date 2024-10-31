from ultralytics import YOLO
from torchsummary import summary
from collections import OrderedDict
import re

def extract_layers(model_str):

    # pattern_with_num = r'\((\d+)\):\s+(\w+)\('
    pattern = r'\((\d+)\):\s*(\w+)'
    # 使用 re.findall 找到所有匹配的编号和层类型
    matches = re.findall(pattern, model_str)
    # 创建一个列表，存储编号和层类型
    layers = []
    for num, layer_type in matches:
        if layer_type not in ["Bottleneck","Sequential"]:
            layers.append(f"{layer_type}")

    return layers

if __name__ == '__main__':
    model = YOLO("D:\\Desktop\\ultralytics\\runs\\detect\\train54\\weights\\best.pt")
    print(extract_layers(str(model)))

    model.val(data="LROC.yaml", split="test", epochs=200, imgsz=640)