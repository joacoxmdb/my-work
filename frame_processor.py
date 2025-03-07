import cv2               # Para abrir y leer el video, redimensionar imágenes
import numpy as np       # Aunque en este módulo se usa menos, es útil para operaciones numéricas si se requiere más adelante


def process_video(video_path, target_fps=5, resize_dim=(640, 480)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / target_fps))
    frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame = cv2.resize(frame, resize_dim)
            frames.append(frame)
        frame_index += 1

    cap.release()
    return frames
