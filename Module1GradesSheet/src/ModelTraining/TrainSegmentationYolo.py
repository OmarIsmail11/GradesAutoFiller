from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='C:/Users/youse/Desktop/University/Image/GradesAutoFiller/Module1GradesSheet/data/SegmentationAnnotations/data.yaml', 
    epochs=100,
    imgsz=640,
    batch=8,
    name='digit_segmenter'
)

print("Training finished.")