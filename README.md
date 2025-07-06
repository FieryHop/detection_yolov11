# detection_yolov11
# Детекция блюд в ресторане с YOLOv11

Проект для автоматического распознавания блюд в ресторанных условиях с использованием YOLOv11.

## 🚀 Быстрый старт

1. Установите зависимости:
```bash
pip install -r requirements.txt
```
Запустите пайплайн обработки и обучения:

```bash
python scripts/run_pipeline.py
```
## 🛠 Полная инструкция
### 1. Подготовка данных
Ручная аннотация (рекомендуется)

Для получения качественных результатов мы используем ручное аннотирование с CVAT:

1. Установите Docker Desktop:
        
    https://www.docker.com/products/docker-desktop
        
2. Запустите CVAT:
```bash
git clone https://github.com/opencv/cvat
cd cvat
docker-compose up -d
docker exec -it cvat_server bash -ic 'python3 manage.py createsuperuser'
```
3. Откройте в браузере: http://localhost:8080
4. Создайте проект:
* Название: "Restaurant Dishes"
* Добавьте метки: pizza, soup, salad, steak, dessert
5. Создайте задачу:
* Название: "Dishes Annotation"
* Загрузите изображения из data/extracted_frames
6. Аннотируйте изображения:
* Используйте инструмент "Rectangle"
* Обводите ТОЛЬКО блюда
* Присваивайте правильные метки
7. Экспортируйте аннотации:
* Формат: YOLO 1.1
* Сохраните файлы с папки obj_train_data в data/annotated_food в проекте


#### Автоматическая аннотация (экспериментальная)
⚠️ Автоматическая аннотация находится в разработке и может требовать ручной корректировки

```bash
python scripts/auto_annotate.py
```
### 2. Обработка данных
* Скачайте yolov11
    ```bash
    python scripts/download_models.py
    ```
* Объедините аннотации:

    ```bash
    python scripts/merge_annotations.py
    ```
* Аугментируйте данные:

    ```bash
    python scripts/augment_dataset.py
    ```
### 3. Обучение модели
```bash
python scripts/train_model.py
```
### 4. Оценка и визуализация
```bash
python scripts/evaluate_model.py
python scripts/visualize_results.py
python scripts/generate_report.py
```
## 📊 Анализ результатов
Результаты будут сохранены в:

outputs/metrics/ - метрики оценки

outputs/predictions/ - примеры предсказаний

outputs/report/ - финальный отчет

## ⏱ Время выполнения
|Этап	|Время (CPU)	|Время (GPU)|
|:-|:-:|:-:|
|Аннотация (100 изображений)	|1-2 часа|	-
|Обучение (100 эпох)	|6-8 часов|	30-60 минут|
|Оценка	|10-15 минут|	2-5 минут|

