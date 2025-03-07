import os                # Para manejo de directorios y archivos
import argparse          # Para parsear argumentos de línea de comando
import cv2               # OpenCV para procesamiento de video e imágenes
import numpy as np       # Operaciones con arreglos, aunque aquí se usa menos directamente

# Importar funciones de los módulos locales
from frame_processor import process_video
from motion_detector import detect_motion
from visualizer import visualize_results

def main():
    print("Inicio del script")  # Depuración
    args = parse_args()
    ...


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sports Motion Detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--fps", type=int, default=5, help="Target frames per second")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    print(f"Procesando video: {args.video}")

    frames = process_video(args.video, args.fps)
    print(f"Se extrajeron {len(frames)} fotogramas")

    motion_results = []
    for i, frame in enumerate(frames):
        print(f"Procesando fotograma {i + 1}/{len(frames)}")
        motion_boxes = detect_motion(frames, i)
        motion_results.append(motion_boxes)

    visualize_results(frames, motion_results, args.output)
    print(f"Procesamiento completo. Resultados guardados en {args.output}")
    
if __name__ == "__main__":
    main()
