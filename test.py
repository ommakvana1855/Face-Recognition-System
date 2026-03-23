from ultralytics import YOLO

# Load PyTorch model
model = YOLO("yolo26x.pt")

# Export to TensorRT engine
model.export(
    format="engine",   # TensorRT
    device=0,          # GPU id
    half=True,         # FP16 (recommended)
    dynamic=True,      # dynamic input shapes
    workspace=4        # GB (adjust based on GPU)
)

print("Export complete: yolo26x.engine")