from ultralytics import YOLO
import cv2
import os

#TODO: Define input and output directories
dataset = "train, val, or test"
activity = "stand, sit, or sleep"

#TODO: Define class index from data.yaml
class_index = 0

raw_dataset = f'{activity}-dataset/{dataset}'
output_image_folder = f'exercise-dataset/{dataset}/images'
output_txt_folder = f'exercise-dataset/{dataset}/labels'

# Initialize the YOLO model
model = YOLO('yolo-Weights/yolov8m-pose.pt')

# Iterate through images in the input folder
for filename in os.listdir(raw_dataset):

    image_path = os.path.join(raw_dataset, filename)

    # Read the image
    img = cv2.imread(image_path)
    
    # Resize image to 640x640
    img = cv2.resize(img, (640, 640))

    # Detect objects
    results = model(source=img, show=False, conf=0.5, stream=False)

    # Annotate and save results
    for i, detection in enumerate(results):
        output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(filename)[0]}.jpg")
        output_txt_path = os.path.join(output_txt_folder, f"{os.path.splitext(filename)[0]}.txt")
        
        # Write annotations to a text file
        with open(output_txt_path, "w") as f:
            if detection.boxes:
                for j in range(len(detection.boxes)):  # Access the class values directly
                    image_width, image_height = detection.orig_shape[1], detection.orig_shape[0]  # Get original image width and height
                    center_x = (detection.boxes.xyxy[j][0] + detection.boxes.xyxy[j][2]) / 2 / image_width  # Normalize center_x
                    center_y = (detection.boxes.xyxy[j][1] + detection.boxes.xyxy[j][3]) / 2 / image_height  # Normalize center_y
                    width = (detection.boxes.xyxy[j][2] - detection.boxes.xyxy[j][0]) / image_width  # Normalize width
                    height = (detection.boxes.xyxy[j][3] - detection.boxes.xyxy[j][1]) / image_height  # Normalize height
                    
                    f.write(f"{class_index} {center_x} {center_y} {width} {height} ")

                    # Extract keypoints if available
                    keypoints = detection.keypoints.xy if detection.keypoints else None
                    if keypoints is not None:
                        for k in range(keypoints.shape[1]):
                            x = keypoints[j, k, 0].item() / image_width  # Normalize x coordinate
                            y = keypoints[j, k, 1].item() / image_height  # Normalize y coordinate
                            visibility = 0 if keypoints[j, k, 0] == 0.0 and keypoints[j, k, 1] == 0.0 else 1 #keypoint visibility 
                            f.write(f"{x} {y} {visibility} ")
                        f.write("\n")

            # Save annotated image
            annotated_image = detection.orig_img
            cv2.imwrite(output_image_path, annotated_image)
            print(f"Processed: {filename}")