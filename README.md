# AI Safety & Collision Warning System 

A real-time computer vision application that detects, tracks, and analyzes objects (people, vehicles, bikes) to provide safety alerts based on proximity and **Time-To-Collision (TTC)**.

##  Overview
This system uses **YOLOv8** for object detection and **ByteTrack** for persistent tracking. It implements a heuristic-based approach to estimate:
* **Real-world Distance:** Calculated based on bounding box height variations.
* **Object Speed:** Estimated by tracking pixel movement over time across frames.
* **TTC (Time-To-Collision):** Predicted time until a potential impact if the current trajectory continues.

##  Risk Levels
The system categorizes safety risks into four visual levels:
*  **NO RISK:** Objects are far or moving away.
*  **CAUTION:** Objects are within the proximity threshold ($<25m$).
*  **DANGER ALERT:** Predicted TTC is less than 3 seconds.
*  **IMMINENT DANGER:** Predicted TTC is less than 0.8 seconds.

##  Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME'''

##  Install dependencies:

Bash
pip install -r requirements.txt
Download the Model:
The script uses yolov8s.pt. It will download automatically on the first run, or you can place it in the root directory.

##  Usage
Update the VIDEO_PATH variable in the script to point to your video file, then run:
Bash
python safety_system.py
Press 'q' to exit the application.

##  Configuration
You can tune the safety parameters at the top of the script:
* **CONFIDENCE_THRESHOLD:** Filter out weak detections (default: 0.5).
* **TTC_WARNING_THRESHOLD_S:** Seconds before a warning is triggered.
* **HEURISTIC_MIN_DIST_M:** Minimum assumed distance in meters.
