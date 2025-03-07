import os                # Para crear directorios y gestionar rutas de archivos
import cv2               # Para escribir imágenes y videos, dibujar cuadros y texto en los frames
import numpy as np       # Si necesitas realizar alguna operación numérica en el futuro


def visualize_results(frames, motion_results, output_dir):
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "motion_detection.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, 5, (width, height))

    for i, frame in enumerate(frames):
        vis_frame = frame.copy()
        if i > 0:
            for box in motion_results[i]:
                x, y, w, h = box
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, vis_frame)
        video_writer.write(vis_frame)

    video_writer.release()
    print(f"Visualización guardada en {video_path}")
    print(f"Fotogramas individuales guardados en {frames_dir}")
