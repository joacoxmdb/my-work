# my-work
Sports Motion Detection
This project is a simple sports motion detection application that processes a short video clip to detect and highlight areas of motion. It uses Python and OpenCV to extract frames from a video, compare consecutive frames for motion detection, and visualize the detected motion by drawing bounding boxes on the frames. The final output is both a video and individual image files with the detected motion.

Features
Video Processing:
Extracts frames from the input video at a specified frame rate (default is 5 fps) and resizes them to a standard resolution (640x480).

Motion Detection:
Compares consecutive frames to detect areas of movement using frame differencing, thresholding, and contour detection. It draws bounding boxes around regions with significant motion.

Visualization:
Outputs a video (motion_detection.mp4) with motion detection results and saves individual processed frames in a folder.

Prerequisites
Ensure you have the following installed:

Python 3.6+
OpenCV: For video and image processing.
NumPy: For array manipulation.
You can install the required Python libraries using pip: "pip install opencv-python numpy"

Project Structure:

sports-motion-detection/
│
├── main.py                 # Main entry point that orchestrates the processing
├── frame_processor.py      # Module for video processing and frame extraction
├── motion_detector.py      # Module for detecting motion between frames
├── visualizer.py           # Module for visualizing and saving results
└── README.md               # Project documentation (this file)

How It Works

Extracting Frames:
The application opens the video file, extracts frames at the target FPS, and resizes them to 640x480 pixels.

Motion Detection:
Each frame (after the first) is compared to its previous frame. The process involves:

Converting frames to grayscale.
Applying a Gaussian blur to reduce noise.
Calculating the absolute difference between consecutive frames.
Thresholding the difference to highlight significant changes.
Dilating the thresholded image to fill in gaps.
Finding contours and drawing bounding boxes around areas that exceed a specified minimum area.
Visualization:
For each processed frame, the app:

Draws bounding boxes around the detected motion regions.
Adds a text overlay showing the frame number.
Saves each processed frame as an image.
Compiles the frames into a final video file.
How to Use
Clone the Repository:

Clone this project to your local machine:
git clone https://github.com/joacoxmdb/my-work.git
cd my-work
Install Dependencies:

Make sure you have the required dependencies installed:

pip install opencv-python numpy
Run the Application:

Use the command line to execute the script. You need to provide the path to the input video file and, optionally, the output directory and target FPS. For example:

python main.py --video my-work/sample_video_clip.mp4 --output output --fps 5
View the Results:

Once the processing is complete, the results will be saved in the output directory specified:

A video file named motion_detection.mp4 with the detection results.
A folder named frames containing individual image files for each processed frame.
Troubleshooting
No Output or Empty Output Directory:

Ensure the video path is correct and the video file exists.
Check if you have the proper permissions to write to the output directory.
Verify that your Python environment has the required dependencies installed.
Error "import cv2 could not be resolved":

Make sure OpenCV is installed in your current Python environment (pip install opencv-python).
Ensure your IDE or text editor is configured to use the correct Python interpreter.
Adjusting Parameters:

If too many or too few motion areas are detected, adjust the threshold and minimum area parameters in motion_detector.py.
Future Improvements
Implement background subtraction techniques for more robust motion detection.
Enhance the algorithm to handle camera movement.
Develop a graphical user interface (GUI) for easier usage.
Optimize the code for real-time processing.