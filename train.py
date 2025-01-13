from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
results = model.train(data='specs.yaml', amp=False, epochs=10)