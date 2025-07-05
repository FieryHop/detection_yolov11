import yaml
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
import pandas as pd


def prepare_dataset(data_config, train_config):
    """Подготовка train/val/test разделов"""
    # Извлекаем параметры разделения
    val_fraction = train_config.get('val_fraction', 0.15)
    test_fraction = train_config.get('test_fraction', 0.1)

    # Создаем директории
    dataset_dir = "dataset"
    os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
    os.makedirs(f"{dataset_dir}/images/test", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/test", exist_ok=True)

    # Собираем все изображения (оригинальные + аугментированные)
    all_images = []

    # Оригинальные изображения
    if os.path.exists(data_config['frames_dir']):
        for img in os.listdir(data_config['frames_dir']):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(('original', img))

    # Аугментированные изображения
    if os.path.exists(data_config.get('augmented_dir', '')):
        for img in os.listdir(data_config['augmented_dir']):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(('augmented', img))

    # Перемешиваем
    random.shuffle(all_images)

    # Разделение данных
    if test_fraction > 0:
        train_val, test = train_test_split(
            all_images,
            test_size=test_fraction,
            random_state=42
        )
        train, val = train_test_split(
            train_val,
            test_size=val_fraction / (1 - test_fraction),
            random_state=42
        )
    else:
        train, val = train_test_split(
            all_images,
            test_size=val_fraction,
            random_state=42
        )
        test = []

    # Функция для копирования данных
    def copy_data(split, data_list):
        for source_type, img in data_list:
            # Определяем исходные пути
            if source_type == 'original':
                img_src = os.path.join(data_config['frames_dir'], img)
                label_src = os.path.join(data_config['annotations_dir'],
                                         os.path.splitext(img)[0] + '.txt')
            else:  # augmented
                img_src = os.path.join(data_config['augmented_dir'], img)
                label_src = os.path.join(data_config['augmented_dir'],
                                         os.path.splitext(img)[0] + '.txt')

            # Копируем изображение
            img_dst = os.path.join(dataset_dir, 'images', split, img)
            shutil.copy(img_src, img_dst)

            # Копируем метку, если существует
            if os.path.exists(label_src):
                label_dst = os.path.join(dataset_dir, 'labels', split,
                                         os.path.splitext(img)[0] + '.txt')
                shutil.copy(label_src, label_dst)
            else:
                # Создаем пустой файл аннотаций
                label_dst = os.path.join(dataset_dir, 'labels', split,
                                         os.path.splitext(img)[0] + '.txt')
                open(label_dst, 'a').close()

    # Копируем данные
    copy_data('train', train)
    copy_data('val', val)
    if test:
        copy_data('test', test)

    # Создаем dataset.yaml
    yaml_content = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test' if test else 'images/val',
        'names': data_config['classes'],
        'nc': len(data_config['classes'])
    }

    with open('dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Dataset prepared: {len(train)} train, {len(val)} val, {len(test)} test images")


def get_model_path(model_name):
    """Возвращает корректный путь к модели"""
    # Проверяем различные возможные пути
    possible_paths = [
        model_name,
        f"models/pretrained/{model_name}",
        f"models/pretrained/{model_name.lower()}",
        f"models/pretrained/{model_name.upper()}",
        f"models/pretrained/yolo{model_name[5:]}",  # Для yolov11s.pt -> v11s.pt
        f"models/pretrained/yolo{model_name[5:].lower()}",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Если модель не найдена, пытаемся скачать
    print(f"Model {model_name} not found. Attempting to download...")
    download_script = os.path.join(os.path.dirname(__file__), "download_models.py")
    if os.path.exists(download_script):
        os.system(f"python {download_script}")
        # Проверяем снова после скачивания
        for path in possible_paths:
            if os.path.exists(path):
                return path

    raise FileNotFoundError(f"Model not found: {model_name}. Please download it manually.")


def train_model(train_config, data_config):
    """Обучает модель YOLOv11 и сохраняет лучшие веса"""
    # Проверяем доступность GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Получаем путь к модели
    model_name = train_config.get('model', 'yolov11s.pt')
    model_path = get_model_path(model_name)
    print(f"Loading model from: {model_path}")

    # Загружаем модель
    model = YOLO(model_path)

    # Параметры обучения
    epochs = train_config.get('epochs', 100)
    imgsz = train_config.get('imgsz', 640)
    batch = train_config.get('batch', 16)

    # Создаем словарь параметров для обучения
    train_args = {
        'data': 'dataset.yaml',
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'optimizer': train_config.get('optimizer', 'auto'),
        'lr0': train_config.get('lr0', 0.01),
        'cos_lr': train_config.get('cos_lr', True),
        'augment': train_config.get('augment', True),
        'device': device,
        'project': 'models/trained',
        'name': 'exp',
        'save_period': 10,
        'patience': train_config.get('patience', 50),
        'dropout': train_config.get('dropout', 0.0),
        'weight_decay': train_config.get('weight_decay', 0.0005),
    }

    # Удаляем неподдерживаемые параметры
    unsupported_args = ['cutmix', 'mixup', 'metrics']
    for arg in unsupported_args:
        if arg in train_args:
            del train_args[arg]

    try:
        # Обучение с основными параметрами
        results = model.train(**train_args)
    except Exception as e:
        print(f"Training failed: {str(e)}")
        print("Trying to reduce batch size...")
        # Пробуем уменьшить batch size
        train_args['batch'] = max(4, batch // 2)
        results = model.train(**train_args)

    # Сохраняем лучшую модель
    best_model_path = f"models/trained/exp/weights/best.pt"
    if os.path.exists(best_model_path):
        os.makedirs('models/trained', exist_ok=True)
        shutil.copy(best_model_path, 'models/trained/best.pt')
        print(f"Best model saved to 'models/trained/best.pt'")

    return results


if __name__ == "__main__":
    # Загрузка конфигов
    with open('configs/data_config.yaml') as f:
        data_config = yaml.safe_load(f)['dataset']
    with open('configs/train_params.yaml') as f:
        train_config = yaml.safe_load(f)

    # Подготовка данных
    prepare_dataset(data_config, train_config)

    # Обучение модели
    results = train_model(train_config, data_config)

    # Сохраняем метрики обучения
    os.makedirs('outputs/metrics', exist_ok=True)
    results_csv = os.path.join('outputs/metrics', 'training_metrics.csv')

    # Сохраняем метрики, если они доступны
    if hasattr(results, 'results_df') and results.results_df is not None:
        results.results_df.to_csv(results_csv, index=False)
        print(f"Training metrics saved to {results_csv}")
    else:
        print("Warning: No training metrics available to save")