from ultralytics import YOLO
import cv2
import os

# Load the image
image_path = "path-to-image"
frame = cv2.imread(image_path)

# Resize the frame to the desired width and height
frame = cv2.resize(frame, (720, 480))

# model
model = YOLO("yolo-Weights/yolov8m-pose.pt")

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
        num_keypoints = keypoint.xy.shape[1]
        print("KP size:", num_keypoints)

        # Loop through each points in the body
        for i in range(num_keypoints):
            x, y = keypoint.xy[0, i, 0].item(), keypoint.xy[0, i, 1].item()

            # Check if keypoint is not (0, 0)
            if x > 0 and y > 0:
                x_norm = x / frame_width
                y_norm = y / frame_height
                cv2.circle(frame, (int(x_norm * frame_width), int(y_norm * frame_height)), 2, (0, 255, 0), -5)


# Define the path to the labeled folder
save_dir = "labeled"
os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn't exist

# Save the annotated frame image
cv2.imwrite(os.path.join(save_dir, "kp-เก็บคาง-2.jpg"), frame)

# Display the frame
cv2.imshow('Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
