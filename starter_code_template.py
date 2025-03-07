# main.py
"""
HomeTeam Network - AI Engineer Take-Home Project
Simple Sports Motion Detection

This is the main entry point for the motion detection program.
"""

import os
import argparse
import cv2
import numpy as np

from frame_processor import process_video
from motion_detector import detect_motion
from visualizer import visualize_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sports Motion Detection")
    parser.add_argument(
        "--video", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--fps", type=int, default=5, help="Target frames per second")
    return parser.parse_args()


def main():
    """Main function to run the motion detection pipeline."""
    # Parse arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    print(f"Processing video: {args.video}")

    # Step 1: Extract frames from video
    frames = process_video(args.video, args.fps)
    print(f"Extracted {len(frames)} frames")

    # Step 2: Detect motion in frames
    motion_results = []
    for i, frame in enumerate(frames):
        print(f"Processing frame {i + 1}/{len(frames)}")

        # Pass the entire frames list and the current index to detect_motion
        # This allows the function to access both current and previous frames
        # for frame comparison and motion detection
        motion_boxes = detect_motion(frames, i)
        motion_results.append(motion_boxes)

    # Step 3: Visualize and save results
    visualize_results(frames, motion_results, args.output)

    print(f"Processing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()

# frame_processor.py
"""
Frame processing functions for the motion detection project.
"""

import cv2
import numpy as np


def process_video(video_path, target_fps=5, resize_dim=(640, 480)):
    """
    Extract frames from a video at a specified frame rate.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second to extract
        resize_dim: Dimensions to resize frames to (width, height)

    Returns:
        List of extracted frames
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval for the target FPS
    frame_interval = max(1, int(original_fps / target_fps))

    # Extract frames
    frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only keep frames at the target frame rate
        if frame_index % frame_interval == 0:
            # Resize frame
            if resize_dim:
                frame = cv2.resize(frame, resize_dim)
            frames.append(frame)

        frame_index += 1

    # Release video capture
    cap.release()

    return frames


# motion_detector.py
"""
Motion detection functions for the sports video analysis project.
"""

import cv2
import numpy as np


def detect_motion(frames, frame_idx, threshold=25, min_area=100):
    """
    Detect motion in the current frame by comparing with previous frame.

    Args:
        frames: List of video frames
        frame_idx: Index of the current frame
        threshold: Threshold for frame difference detection
        min_area: Minimum contour area to consider

    Returns:
        List of bounding boxes for detected motion regions
    """
    # We need at least 2 frames to detect motion
    if frame_idx < 1 or frame_idx >= len(frames):
        return []

    # Get current and previous frame
    current_frame = frames[frame_idx]
    prev_frame = frames[frame_idx - 1]

    # Convert frames to grayscale
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    current_blur = cv2.GaussianBlur(current_gray, (5, 5), 0)
    prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(prev_blur, current_blur)

    # Apply threshold to highlight differences
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and get bounding boxes
    motion_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        motion_boxes.append((x, y, w, h))

    return motion_boxes


# visualizer.py
"""
Visualization functions for displaying motion detection results.
"""

import os
import cv2
import numpy as np


def visualize_results(frames, motion_results, output_dir):
    """
    Create visualization of motion detection results.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        output_dir: Directory to save visualization results
    """
    # Create output directory for frames
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Get dimensions for the output video
    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "motion_detection.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, 5, (width, height))

    # Process each frame
    for i, frame in enumerate(frames):
        # Create a copy for visualization
        vis_frame = frame.copy()

        # Draw bounding boxes for motion regions
        if i > 0:  # Skip the first frame as it has no motion data
            for box in motion_results[i]:
                x, y, w, h = box
                # Draw rectangle around motion region
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add frame number
        cv2.putText(
            vis_frame,
            f"Frame: {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Save frame as image
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, vis_frame)

        # Write frame to video
        video_writer.write(vis_frame)

    # Release video writer
    video_writer.release()

    print(f"Visualization saved to {video_path}")
    print(f"Individual frames saved to {frames_dir}")
