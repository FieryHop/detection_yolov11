import os
import requests
from tqdm import tqdm

MODELS = {
    "yolov11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "yolov11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolov11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    "yolov11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
}


def download_file(url, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as f, tqdm(
            desc=os.path.basename(filename),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


if __name__ == "__main__":
    print("Downloading YOLOv11 models...")
    for name, url in MODELS.items():
        output_path = f"models/pretrained/{name}"
        if not os.path.exists(output_path):
            download_file(url, output_path)
        else:
            print(f"{name} already exists, skipping download")

    print("All models downloaded to models/pretrained/")