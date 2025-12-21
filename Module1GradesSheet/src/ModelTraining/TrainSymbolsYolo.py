from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='C:/Users/youse/Desktop/University/Image/GradesAutoFiller/Module1GradesSheet/data/Symbols-2.v2-symbolsdata.yolov12/data.yaml', 
    epochs=100, 
    imgsz=640, 
    batch=16, 
    scale=0.5,
    mosaic=1.0,
    fliplr=0.0,
    flipud=0.0,
    hsv_s=0.5,
    hsv_v=0.4,
)

print("Training finished.")