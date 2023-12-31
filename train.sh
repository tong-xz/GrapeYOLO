pip install ultralytics
yolo task=detect mode=train model=yolov8m.pt data=./GrapeYOLO-2/data.yaml epochs=30 imgsz=640