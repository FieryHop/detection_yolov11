from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np


def extract_metric(value):
    """Извлекает скалярное значение из метрики, которая может быть массивом"""
    if isinstance(value, (list, np.ndarray)):
        return float(value[0])  # Берем первый элемент массива
    return float(value)


def evaluate_model():
    """Оценка модели и сохранение метрик"""
    # Проверяем доступность лучшей модели
    model_path = "models/trained/best.pt"
    if not os.path.exists(model_path):
        model_path = "models/trained/exp/weights/best.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Best model not found. Please run training first.")

    model = YOLO(model_path)

    # Оценка модели на тестовом наборе
    results = model.val(
        data='dataset.yaml',
        split='test',
        batch=8
    )

    # Извлекаем метрики с учетом возможной структуры данных
    metrics = {
        'mAP50': extract_metric(results.box.map50),
        'mAP50-95': extract_metric(results.box.map),
        'precision': extract_metric(results.box.mp),
        'recall': extract_metric(results.box.mr),
        'f1': extract_metric(results.box.f1)
    }

    # Создаем директорию для метрик
    os.makedirs('outputs/metrics', exist_ok=True)

    # Сохраняем метрики в файл
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('outputs/metrics/evaluation_metrics.csv', index=False)

    # Визуализация confusion matrix
    confusion_matrix = results.confusion_matrix.matrix
    class_names = list(model.names.values())

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt='g',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('outputs/metrics/confusion_matrix.png', dpi=300)
    plt.close()

    # Дополнительная визуализация метрик
    plt.figure(figsize=(10, 6))

    # Преобразуем метрики в списки
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # Создаем bar plot
    bars = plt.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom',
                 fontsize=10,
                 fontweight='bold')

    plt.title('Model Evaluation Metrics', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем горизонтальную линию для идеального значения
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)

    plt.savefig('outputs/metrics/metrics_barplot.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("Evaluation completed. Metrics saved to outputs/metrics/")
    print(
        f"Test results: mAP50={metrics['mAP50']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")


if __name__ == "__main__":
    evaluate_model()