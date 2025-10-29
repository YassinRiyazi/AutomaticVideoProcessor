import os
import tensorrt as trt
from ultralytics import YOLO

def engine_maker(name: str):
    """
    Rebuilds a YOLO .engine file from a .pt model using Ultralytics.
    Automatically removes intermediate .onnx file.
    """
    BaseAddress = os.path.abspath(os.path.dirname(__file__))
    BaseAddress = os.path.join(BaseAddress, "BaseUtils", "Detection", "Weights")
    
    model_path = os.path.join(BaseAddress, f"{name}.pt")
    engine_path = os.path.join(BaseAddress, f"{name}.engine")
    onnx_path = os.path.join(BaseAddress, f"{name}.onnx")

    print(f"[INFO] Building TensorRT engine for: {name}")
    model = YOLO(model_path)
    model.export(
        format="engine",
        imgsz=(640, 640),
        dynamic=False,
        batch=1,
        verbose=False,
        simplify=True,
    )

    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    print(f"[SUCCESS] Exported: {engine_path}")

def get_engine_trt_version(engine_path: str) -> str | None:
    """
    Reads the TensorRT version used to build the given .engine file.

    Args:
        engine_path (str): Path to the TensorRT engine file.

    Returns:
        str | None: The version string (e.g. '10.1.0') if found, else None.
    """
    if not os.path.exists(engine_path):
        return None

    try:
        with open(engine_path, "rb") as f:
            data = f.read(8192)  # Read header portion
            text = data.decode("latin-1", errors="ignore")

        # TensorRT version is typically embedded as "TensorRT x.y.z"
        import re
        match = re.search(r"TensorRT\s+(\d+\.\d+(?:\.\d+)?)", text)
        return match.group(1) if match else None
    except Exception as e:
        print(f"[WARN] Could not read TensorRT version from {engine_path}: {e}")
        return None

def ensure_engine_compatibility(model_names: list[str]):
    """
    Ensures that all .engine files are compatible with the installed TensorRT version.
    If mismatched or missing, rebuilds them automatically using engine_maker().

    Args:
        model_names (list[str]): List of YOLO model base names (without extension).
    """
    installed_trt_version = trt.__version__
    print(f"[INFO] Installed TensorRT version: {installed_trt_version}")

    base_path = os.path.abspath(os.path.dirname(__file__))

    for name in model_names:
        engine_path = os.path.join(base_path, f"{name}.engine")
        engine_version = get_engine_trt_version(engine_path)

        if not os.path.exists(engine_path):
            print(f"[WARN] Missing engine file: {engine_path}")
            engine_maker(name)
            continue

        if engine_version is None:
            print(f"[WARN] Could not determine TensorRT version for {engine_path}")
            engine_maker(name)
            continue

        if engine_version != installed_trt_version:
            print(f"[WARN] Mismatch detected for {name}:")
            print(f"       Engine built with TensorRT {engine_version}")
            print(f"       Installed TensorRT {installed_trt_version}")
            print(f"[ACTION] Rebuilding engine for {name}...")
            engine_maker(name)
        else:
            print(f"[OK] {name}.engine is compatible (TensorRT {engine_version})")

# checking whether ffmpeg and ffprobe is in the PATH
def check_ffmpeg_ffprobe():
    from shutil import which

    if which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg not found in PATH. Please install ffmpeg and add it to PATH.")
    if which("ffprobe") is None:
        raise EnvironmentError("ffprobe not found in PATH. Please install ffprobe and add it to PATH.")

    print("[INFO] ffmpeg and ffprobe are available in PATH.")

if __name__ == "__main__":
    ensure_engine_compatibility(["Gray-320-s", "Gray-320-n"])
    check_ffmpeg_ffprobe()