import os
import shutil


def merge_annotations(auto_dir, manual_dir, output_dir):
    """
    Объединяет автоматические и ручные аннотации
    """
    os.makedirs(output_dir, exist_ok=True)

    # Копируем все ручные аннотации
    for file in os.listdir(manual_dir):
        if file.endswith('.txt'):
            src = os.path.join(manual_dir, file)
            dst = os.path.join(output_dir, file)
            shutil.copy(src, dst)

    # Добавляем автоматические аннотации, где нет ручных
    for file in os.listdir(auto_dir):
        if file.endswith('.txt'):
            dst = os.path.join(output_dir, file)
            if not os.path.exists(dst):
                src = os.path.join(auto_dir, file)
                shutil.copy(src, dst)


if __name__ == "__main__":
    merge_annotations(
        auto_dir="data/annotated_food",
        manual_dir="data/manual_annotations",
        output_dir="data/final_annotations"
    )