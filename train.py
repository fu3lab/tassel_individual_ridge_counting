from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data=r"E:\AgRobot\code\yolov8\ultralytics-main/tassel_data.yaml", epochs=100, workers=0, batch=2,imgsz=1280
            ,amp=True)  # train the model
metrics = model.val()  # evaluate model performance on the validation set


