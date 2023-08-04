from ultralytics import YOLO

if __name__ == "__main__" :
    model = YOLO("yolov8m.pt")
    model.train(data="car_data.yaml", epochs=100, batch=64, lrf=0.025)

    # model = YOLO("./runs/detect/train/weights/last.pt")
    # model.train(resume=True)