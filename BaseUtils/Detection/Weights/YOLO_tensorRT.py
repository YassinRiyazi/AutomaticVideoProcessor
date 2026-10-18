"""
    Author: Yassin Riyazi
    Date: 01-07-2025
    Description: This script exports YOLO models to TensorRT format. 
    YOLO models are trained by Yassin Riyazi for contact angle detection.
    Usage: Run this script to export the models.

"""
import os
from ultralytics import YOLO


def enginemaker(name: str = "Gray-320-s"):
    BaseAddress = os.path.abspath(os.path.dirname(__file__))
    model = YOLO(os.path.join(BaseAddress, f"{name}.pt"))

    # Export the model to TensorRT format
    model.export(format="engine",
                imgsz=(640, 640),
                dynamic=False,
                #  int8=True,
                batch=1,
                #  half=True,
                verbose=False,
                simplify=True)

    os.remove(os.path.join(BaseAddress, f"{name}.onnx"))

# Load the YOLO11 model
enginemaker("Gray-320-s")

enginemaker("Gray-320-n")