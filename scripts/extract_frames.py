import cv2
import os
from tqdm import tqdm


def extract_frames(video_dir, output_dir, frame_interval=10):
    """
    Извлекает кадры из видео с интервалом
    :param video_dir: Папка с исходными видео
    :param output_dir: Папка для сохранения кадров
    :param frame_interval: Интервал извлечения (кадры)
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))]

    for video_file in tqdm(video_files, desc="Processing videos"):
        cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_name = f"{os.path.splitext(video_file)[0]}_frame{saved_count:04d}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_name), frame)
                saved_count += 1

            frame_count += 1

        cap.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', default='data/raw_videos', help='Input video directory')
    parser.add_argument('--output-dir', default='data/extracted_frames', help='Output frames directory')
    parser.add_argument('--interval', type=int, default=10, help='Frame extraction interval')
    args = parser.parse_args()

    extract_frames(args.video_dir, args.output_dir, args.interval)