from ultralytics import YOLO
import cv2
import numpy as np

def calculate_distance(lpt1, rpt1, lpt2, rpt2):
    left_distance = np.sqrt((lpt1[0] - lpt2[0]) ** 2 + (lpt1[1] - lpt2[1]) ** 2)
    right_distance = np.sqrt((rpt1[0] - rpt2[0]) ** 2 + (rpt1[1] - rpt2[1]) ** 2)
    return np.mean([left_distance, right_distance])

def calculate_angle(point1, point2, point3):
    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]], dtype=float)
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]], dtype=float)
    vector1 /= np.linalg.norm(vector1)
    vector2 /= np.linalg.norm(vector2)
    dot_product = np.clip(np.dot(vector1, vector2), -1, 1)
    angle = np.degrees(np.arccos(dot_product))
    return angle

def stand_to_sit(curr, stand):
    return curr is not None and stand is not None and (curr - stand) < -10

def sit_to_stand(curr, sit):
    return curr is not None and sit is not None and (curr - sit) > 10

def sit_to_sleep(angle, sleep):
    if angle is not None: 
        if sleep is None:
            return angle < 85
        else:
            return (angle - sleep) < -10

def sleep_to_sit(angle, sit):
    return angle is not None and sit is not None and (angle - sit) > 10

def set_distance(distance, activity, conf, left_hip, right_hip, left_knee, right_knee):
    if distance is None and conf >= 0.9:
        if activity in [0, 2]:
            distance = calculate_distance(left_hip, right_hip, left_knee, right_knee)
    return distance

def main():

    stand = None
    sit = None
    sleep = None

    model = YOLO('activity-model.pt')

    video_path = "demo.mp4"

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():

        success, frame = cap.read()

        if success:

            results = model.track(frame, persist=True)

            # Get the boxes and track IDs
            # TODO:
            boxes = results[0].boxes.xywh.cpu()
            kps = results[0].keypoints

            # if results[0].boxes.id is not None and len(results[0].boxes.id) > 0:
            #     track_id = results[0].boxes.id[0].item()
            #     print(track_id)

            if kps.shape[1] > 0:
                
                activity = results[0].boxes.cls.cpu().numpy().astype(int)[0]
                conf = results[0].boxes.conf.cpu().numpy().astype(float)[0]

                left_shoulder = (int(kps.xy[0, 5, 0].item()), int(kps.xy[0, 5, 1].item()))
                right_shoulder = (int(kps.xy[0, 6, 0].item()), int(kps.xy[0, 6, 1].item()))
                left_hip = (int(kps.xy[0, 11, 0].item()), int(kps.xy[0, 11, 1].item()))
                right_hip = (int(kps.xy[0, 12, 0].item()), int(kps.xy[0, 12, 1].item()))
                left_knee = (int(kps.xy[0, 13, 0].item()), int(kps.xy[0, 13, 1].item()))
                right_knee = (int(kps.xy[0, 14, 0].item()), int(kps.xy[0, 14, 1].item()))

                stand = calculate_distance(left_hip, right_hip, left_knee, right_knee) if activity == 0 and conf >= 0.9 and stand is None else stand
                sit = calculate_distance(left_hip, right_hip, left_knee, right_knee) if activity == 2 and conf >= 0.9 and sit is None else sit
                sleep = calculate_angle(left_shoulder, left_hip, right_hip) if activity == 1 and conf >= 0.4 and sleep is None else sleep
                print(sleep)

                # Calculate angles between joints (e.g., thigh-torso angle)
                angle = calculate_angle(left_shoulder, left_hip, right_hip)
                print(angle)

                # Calculate current distance between hips and knees
                curr = calculate_distance(left_hip, right_hip, left_knee, right_knee)

                # Detect transition
                if sleep_to_sit(angle, sit) and activity in [2, 1]:
                    cv2.putText(frame, "Sleep to Sit Transition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif sit_to_sleep(angle, sleep) and activity == 2 or (activity == 1 and conf <= 0.4):
                    cv2.putText(frame, "Sit to Sleep Transition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                elif stand_to_sit(curr, stand) and activity in [0, 2] and conf <= 0.9 and angle > 85:
                    cv2.putText(frame, "Stand to Sit Transition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif sit_to_stand(curr, sit) and activity in [0, 2] and conf <= 0.9 and angle > 85:
                    cv2.putText(frame, "Sit to Stand Transition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Transitions Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("Activity Detection", annotated_frame)

            if cv2.waitKey(1) == 27:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


