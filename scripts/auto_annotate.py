import os
import yaml
from ultralytics import YOLO
from tqdm import tqdm


def auto_annotate(config_path):
    """
    Автоматическое аннотирование ТОЛЬКО для классов еды
    :param config_path: Путь к конфиг-файлу
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)['dataset']

    # Используем специализированную модель для детекции еды
    model_path = "models/pretrained/yolov11s.pt"
    if not os.path.exists(model_path):
        # Если модель не скачана - скачиваем
        model = YOLO("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt")
        model.export(format="pt", name=model_path)
    else:
        model = YOLO(model_path)

    frames_dir = config['frames_dir']
    os.makedirs(config['annotations_dir'], exist_ok=True)

    # Фильтрация ТОЛЬКО для классов еды
    food_classes = config.get('food_classes', [])

    for img_file in tqdm(os.listdir(frames_dir), desc="Annotating FOOD"):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(frames_dir, img_file)
        results = model.predict(img_path, conf=0.5)

        # Сохранение аннотаций в формате YOLO
        txt_path = os.path.join(config['annotations_dir'], os.path.splitext(img_file)[0] + '.txt')

        with open(txt_path, 'w') as f:
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    cls_name = model.names[cls_id]

                    # Фильтрация ТОЛЬКО для классов еды
                    if cls_name.lower() in food_classes:
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":
    auto_annotate("configs/data_config.yaml")
