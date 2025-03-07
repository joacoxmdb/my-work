import cv2               # Para convertir a escala de grises, aplicar desenfoque, calcular la diferencia entre frames y detectar contornos
import numpy as np       # Para operaciones con arrays y manejo de datos de imagen

def detect_motion(frames, frame_idx, threshold=25, min_area=100):
    if frame_idx < 1 or frame_idx >= len(frames):
        return []
    
    current_frame = frames[frame_idx]
    prev_frame = frames[frame_idx - 1]

    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    current_blur = cv2.GaussianBlur(current_gray, (5, 5), 0)
    prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    frame_diff = cv2.absdiff(prev_blur, current_blur)
    _, thresh_img = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh_img, None, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        motion_boxes.append((x, y, w, h))
    return motion_boxes
