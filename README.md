# Elderly Activity Monitoring System
This project is a activity detection and tracking system for the elderly, employing a custom trained computer vision model and transition algorithm. It utilizes web camera footage processed by Ultralytics YOLOv8 for object detection and tracking. The system trains a custom model using pose estimation for activity recognition, with results stored in a Firebase database, accessible via a Flutter web app developed in Flutterflow.

## Features
- **Activity Recognition**: Detects activities such as standing, sitting, and sleeping.
  - Pose mAP50 is 0.926, with high accuracy for standing (0.995), sitting (0.942), and moderate for sleeping (0.841).
  - Link to Google Colab: [YOLOv8 Custom Pose Estimation](https://colab.research.google.com/drive/1KIdEeraimaSSi6q0DvRHfmBwo2SIXseF?usp=sharing)
- **Transition Tracking**: Monitors transitions between activities.
- **Custom-Trained Model**: Utilizes a custom-trained computer vision model for accurate activity recognition (activity-model.pt)
- **Memory Monitoring**: Logs and checks the current CPU and memory usage whether it exceeds a specified threshold (default 95%)
- **Camera Availability**: Iterates through a list of camera to find the first available one
- **Dataset Annotation**: Includes a program to annotate datasets according to YOLOv8 pose estimation dataset configuration (ex. activity-dataset.zip).
- **Pose Estimation**: Simple program that uses pose estimation on images and videos.
- **Web Camera Check**: Simple web camera check program.

## Activity Detection Examples
See more evaluation results in the `runs` folder or the [YOLOv8 Custom Pose Estimation](https://colab.research.google.com/drive/1KIdEeraimaSSi6q0DvRHfmBwo2SIXseF?usp=sharing)

<table>
  <tr>
    <td><img src="examples/stand-example.png" alt="standing" style="height: 200px;"></td>
    <td><img src="examples/sit-example.png" alt="sitting" style="height: 200px;"></td>
    <td><img src="examples/sleep-example.png" alt="sleeping" style="height: 200px;"></td>
  </tr>
</table>

## Prerequisites

Make sure you have the following installed:

- Python 3.7 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)
- Optional: [Firebase project](https://firebase.google.com/)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies:**

   ```bash
   pip install opencv-python firebase-admin python-dotenv numpy ultralytics
   ```

3. **Set up your environment variables (optional):**

   Create a `.env` file in the project root directory with the following content and uncomment the Firebase code in the `activit-detection` files

   ```env
   ELDERLY_KEY=path/to/your/firebase/serviceAccountKey.json
   DOC_ID=your_document_id
   ```

## Usage

1. **Run the program:**

   Execute the script using Python:

   ```bash
   python activity-detect-cam.py
   ```

2. **Controls:**

   - The program will start your webcam and begin detecting and tracking activities.
   - If you implement a Firebase datbase, the program will push activity logs to the database every 10 seconds (can edit by changing **frequency** variable).
   - Press `q` to quit the program.

## Notes

- Ensure that your webcam is properly connected and functional.
- The YOLO model  (`activity-model.pt-v8x`) may need fine-tuning for different environments or scenarios.

## Dependencies

- `collections`
- `os`
- `psutil`
- `time`
- `logging`
- `dotenv`
- `opencv-python`
- `firebase-admin`
- `python-dotenv`
- `screeninfo`
- `numpy`
- `ultralytics`

## Troubleshooting

- If you encounter issues with Firebase initialization, ensure the `ELDERLY_KEY` path and `DOC_ID` in the `.env` file are correct.
- For video capture issues, confirm that your webcam is accessible and not being used by other applications.
- Make sure you'll imported the required dependencies.



  
