import os
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_predictions(image, boxes, class_names, conf_threshold=0.5):
    """
    Рисует bounding boxes на изображении
    """
    # Цвета для разных классов
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    colors = (colors[:, :3] * 255).astype(int).tolist()

    for box in boxes:
        if box.conf.item() < conf_threshold:
            continue

        # Координаты bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Информация о классе
        cls_id = int(box.cls)
        cls_name = class_names[cls_id]
        conf = box.conf.item()

        # Выбираем цвет для класса
        color = colors[cls_id % len(colors)]

        # Рисуем прямоугольник
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Подпись с классом и уверенностью
        label = f"{cls_name}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image


def generate_video_predictions(model, video_path, output_path, class_names, conf_threshold=0.5):
    """
    Генерирует видео с предсказаниями
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Создаем VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Делаем предсказание
        results = model(frame, verbose=False)

        # Визуализируем результаты
        if results[0].boxes is not None:
            frame = plot_predictions(
                frame,
                results[0].boxes,
                class_names,
                conf_threshold
            )

        # Записываем кадр
        out.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


def visualize_results():
    """
    Основная функция визуализации
    """
    # Загрузка конфигов
    with open('configs/data_config.yaml') as f:
        data_config = yaml.safe_load(f)['dataset']

    # Загрузка модели
    model = YOLO("models/trained/best.pt")

    # Создаем директорию для результатов
    os.makedirs('outputs/predictions', exist_ok=True)

    # Визуализация на тестовых изображениях
    test_images_dir = os.path.join('dataset', 'images', 'test')
    for img_file in tqdm(os.listdir(test_images_dir), desc="Processing test images"):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(test_images_dir, img_file)
        img = cv2.imread(img_path)

        # Предсказание
        results = model(img, conf=0.5)

        # Визуализация
        if results[0].boxes is not None:
            img = plot_predictions(
                img,
                results[0].boxes,
                data_config['classes']
            )

        # Сохранение
        output_path = os.path.join('outputs/predictions', img_file)
        cv2.imwrite(output_path, img)

    # Обработка видео (если есть)
    raw_videos_dir = data_config['raw_videos_dir']
    if os.path.exists(raw_videos_dir):
        for video_file in os.listdir(raw_videos_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(raw_videos_dir, video_file)
                output_path = os.path.join(
                    'outputs/predictions',
                    f"pred_{os.path.splitext(video_file)[0]}.mp4"
                )
                print(f"Processing video: {video_file}")
                generate_video_predictions(
                    model,
                    video_path,
                    output_path,
                    data_config['classes']
                )


if __name__ == "__main__":
    visualize_results()