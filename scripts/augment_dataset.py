import albumentations as A
import cv2
import os
import yaml
from tqdm import tqdm
import numpy as np
import shutil


def augment_dataset(config_path):
    """
    Упрощенная аугментация данных (только изображения)
    :param config_path: Путь к конфиг-файлу
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)['dataset']
        aug_config = config['augmentation']

    if not aug_config['enabled']:
        print("Augmentation is disabled in config. Skipping...")
        return

    # Создаем трансформации ТОЛЬКО для изображений
    transforms = A.Compose([
        getattr(A, name)(**params)
        for transform in aug_config['transforms']
        for name, params in transform.items()
    ])

    # Создаем директории
    os.makedirs(config['augmented_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['augmented_dir'], 'images'), exist_ok=True)
    os.makedirs(os.path.join(config['augmented_dir'], 'labels'), exist_ok=True)

    # Собираем все изображения
    image_files = [f for f in os.listdir(config['frames_dir'])
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Счетчики для статистики
    total_images = len(image_files)
    processed = 0
    errors = 0

    for img_file in tqdm(image_files, desc="Augmenting"):
        img_path = os.path.join(config['frames_dir'], img_file)
        label_path = os.path.join(config['annotations_dir'], os.path.splitext(img_file)[0] + '.txt')

        # Читаем изображение
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image: {img_file}, skipping...")
            errors += 1
            continue

        # Применяем аугментации
        for i in range(aug_config['augment_count']):
            try:
                # Применяем трансформации только к изображению
                augmented = transforms(image=image)
                aug_img = augmented['image']

                # Генерируем имя для аугментированного изображения
                aug_img_name = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
                img_save_path = os.path.join(config['augmented_dir'], 'images', aug_img_name)

                # Сохраняем изображение
                cv2.imwrite(img_save_path, aug_img)

                # Копируем оригинальные аннотации
                txt_save_path = os.path.join(config['augmented_dir'], 'labels',
                                             os.path.splitext(aug_img_name)[0] + '.txt')
                if os.path.exists(label_path):
                    shutil.copy(label_path, txt_save_path)
                else:
                    # Создаем пустой файл аннотаций
                    open(txt_save_path, 'w').close()

                processed += 1

            except Exception as e:
                print(f"Error augmenting {img_file} (aug{i}): {str(e)}")
                errors += 1

    # Выводим статистику
    print("\nAugmentation completed!")
    print(f"Total images: {total_images}")
    print(f"Augmentations per image: {aug_config['augment_count']}")
    print(f"Total processed augmentations: {processed}")
    print(f"Total errors: {errors}")


if __name__ == "__main__":
    augment_dataset("configs/data_config.yaml")