from ultralytics import YOLO
import cv2
import os

# Load the image
image_path = "dataset/multi-1.jpg"
frame = cv2.imread(image_path)

# model
model = YOLO("yolo-Weights/yolov8s-pose.pt")

# Make a prediction
preds = model(frame)

# Get the keypoints
for pred in preds:
    print("====================================")
    print(pred)
    keypoints = pred.keypoints

    frame_height, frame_width, _ = frame.shape

    # Draw the keypoints on the frame
    for keypoint in keypoints:
        print("keypoint:",keypoint.xy[0])
        # Draw a circle on the frame at the keypoint location 
        # check size of tensor
        print(keypoint.xy.shape[1])

        # loop through each points in the body
        for i in range(keypoint.xy.shape[1]):
            x_norm = keypoint.xy[0, i, 0].item() / frame_width
            y_norm = keypoint.xy[0, i, 1].item() / frame_height
            cv2.circle(frame, (int(x_norm * frame_width), int(y_norm * frame_height)), 10, (0, 255, 0), -5)


# Define the path to the labeled folder
save_dir = "labeled"
os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn't exist

# Save the annotated frame image
cv2.imwrite(os.path.join(save_dir, "stand-19-test.jpg"), frame)

# Display the frame
cv2.imshow('Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
