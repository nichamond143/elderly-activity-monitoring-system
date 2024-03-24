from ultralytics import YOLO
import cv2
import os
import math 

cap = cv2.VideoCapture(0) #
# cap.set(3, 640)
# cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8s-pose.pt")

# object classes
classNames = model.model.names
print

while True:
    success, frame = cap.read()
    if success:


        # Make a prediction
        preds = model(frame)

        #annotate_frme = preds[0].plot()
        # Get the keypoints
        for pred in preds:
            print("====================================")
            print(pred)
            keypoints = pred.keypoints

            # Draw the keypoints on the frame
            for keypoint in keypoints:
                print("keypoint:",keypoint.xy[0])
                #Draw a circle on the frame at the keypoint location 
                # check size of tensor
                print(keypoint.xy.shape[1])

                # loop through each points in the body
                # for i in range(keypoint.xy.shape[1]):
                #     print(i)
                #     cv2.circle(frame, (int(keypoint.xy[0,i,0].item()), int(keypoint.xy[0,i,1].item())), 5, (0, 255, 0), -5)

                # manually draw the points that are in the body
                if(keypoint.xy.shape[1] == 17):
                    #nose
                    cv2.circle(frame, (int(keypoint.xy[0,0,0].item()), int(keypoint.xy[0,0,1].item())), 5, (0, 255, 0), -5)
                    #left eye
                    cv2.circle(frame, (int(keypoint.xy[0,1,0].item()), int(keypoint.xy[0,1,1].item())), 5, (0, 255, 0), -5)
                    #right eye
                    cv2.circle(frame, (int(keypoint.xy[0,2,0].item()), int(keypoint.xy[0,2,1].item())), 5, (0, 255, 0), -5)
                    #left ear
                    cv2.circle(frame, (int(keypoint.xy[0,3,0].item()), int(keypoint.xy[0,3,1].item())), 5, (0, 255, 0), -5)
                    #right ear
                    cv2.circle(frame, (int(keypoint.xy[0,4,0].item()), int(keypoint.xy[0,4,1].item())), 5, (0, 255, 0), -5)
                    #left shoulder
                    cv2.circle(frame, (int(keypoint.xy[0,5,0].item()), int(keypoint.xy[0,5,1].item())), 5, (0, 255, 0), -5)

                    #right shoulder
                    cv2.circle(frame, (int(keypoint.xy[0,6,0].item()), int(keypoint.xy[0,6,1].item())), 5, (0, 255, 0), -5)

                    #left elbow
                    cv2.circle(frame, (int(keypoint.xy[0,7,0].item()), int(keypoint.xy[0,7,1].item())), 5, (0, 255, 0), -5)

                    #right elbow
                    cv2.circle(frame, (int(keypoint.xy[0,8,0].item()), int(keypoint.xy[0,8,1].item())), 5, (0, 255, 0), -5)

                    #left wrist
                    cv2.circle(frame, (int(keypoint.xy[0,9,0].item()), int(keypoint.xy[0,9,1].item())), 5, (0, 4, 255), -5)

                    #right wrist
                    cv2.circle(frame, (int(keypoint.xy[0,10,0].item()), int(keypoint.xy[0,10,1].item())), 5, (0, 4, 255), -5)

                    #left hip
                    cv2.circle(frame, (int(keypoint.xy[0,11,0].item()), int(keypoint.xy[0,11,1].item())), 5, (162, 255, 0), -5)

                    #right hip
                    cv2.circle(frame, (int(keypoint.xy[0,12,0].item()), int(keypoint.xy[0,12,1].item())), 5, (162, 255, 0), -5)

                    #left knee
                    cv2.circle(frame, (int(keypoint.xy[0,13,0].item()), int(keypoint.xy[0,13,1].item())), 5, (255, 155, 0), -5)

                    #right knee
                    cv2.circle(frame, (int(keypoint.xy[0,14,0].item()), int(keypoint.xy[0,14,1].item())), 5, (255, 155, 0), -5)

                    #left ankle
                    cv2.circle(frame, (int(keypoint.xy[0,15,0].item()), int(keypoint.xy[0,15,1].item())), 5, (255, 0, 0 ), -5)

                    #right ankle
                    cv2.circle(frame, (int(keypoint.xy[0,16,0].item()), int(keypoint.xy[0,16,1].item())), 5, (255, 0, 0 ), -5)
            
        # Display the frame
        cv2.imshow('Frame', frame)

        # If the user presses the 'q' key, break out of the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Otherwise, break out of the loop
    else:
        break

cap.release()
cv2.destroyAllWindows()