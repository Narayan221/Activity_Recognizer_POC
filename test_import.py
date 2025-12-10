try:
    print("Importing ultralytics...")
    from ultralytics import YOLO
    print("Import successful. Loading model...")
    model = YOLO('yolov8n-pose.pt')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
