from ultralytics import YOLO

# yolo model creation
model = YOLO("yolo-weights/yolov8l.pt")
model.train(data="coco128.yaml", imgsz=640, batch=4, epochs=10, workers=0)