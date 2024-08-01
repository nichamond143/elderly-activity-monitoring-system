# Elderly Activity Monitoring System
This project is a activity detection and tracking system for the elderly, employing a custom trained computer vision model and transition algorithm. It utilizes web camera footage processed by Ultralytics YOLOv8 for object detection and tracking. The system trains a custom model using pose estimation for activity recognition, with results stored in a Firebase database, accessible via a Flutter web app developed in Flutterflow.

## Features
- **Activity Recognition**: Detects activities such as standing, sitting, and sleeping.
- **Transition Tracking**: Monitors transitions between activities.
- **Custom-Trained Model**: Utilizes a custom-trained computer vision model for accurate activity recognition (activity-model.pt)
- **Dataset Annotation**: Includes a program to annotate datasets according to YOLOv8 pose estimation dataset configuration (ex. activity-dataset.zip).
- **Pose Estimation**: Simple program that uses pose estimation on images and videos.
- **Web Camera Check**: Simple web camera check program.

  
