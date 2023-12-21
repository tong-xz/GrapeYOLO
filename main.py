from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Use the model
    model.train(data="config.yaml", epochs=20)  # train the model
    model.eval(data="config.yaml")
