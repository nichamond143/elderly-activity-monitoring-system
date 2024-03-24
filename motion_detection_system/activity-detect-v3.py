from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO('activity-model.pt')

video_path = "videos/demo-1.mp4"
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: []) 

while cap.isOpened():

    success, frame = cap.read()

    if success:

        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:

            # Get boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            conf = results[0].boxes.conf.cpu().numpy().astype(float)[0]

            # Plot tracks
            for box, track_id in zip(boxes, track_ids):

                x, y, w, h = box

                track = track_history[track_id]
                track.append((float(x), float(y), float(w), float(h)))

                if len(track) >= 10:
                    x_diff = track[-1][0] - track[-10][0]
                    y_diff = track[-1][1] - track[-10][1]
                    w_diff = track[-1][2] - track[-10][2]
                    h_diff = track[-1][3] - track[-10][3]
                     
                    cv2.putText(frame, f"x_diff: {x_diff:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    cv2.putText(frame, f"y_diff: {y_diff:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    cv2.putText(frame, f"w_diff: {w_diff:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    cv2.putText(frame, f"h_diff: {h_diff:.3f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    if abs(y_diff) > 10 or abs(h_diff) > 10 or abs(w_diff) > 10 :
                        if h > w:
                            if h_diff < -15 :
                                cv2.putText(frame, "Stand to Sit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif h_diff > 15 :
                                cv2.putText(frame, "Sit to Stand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, "No Transition Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        else:
                            if w_diff > 10 :
                                cv2.putText(frame, "Sit to Sleep", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            elif w_diff < -10 :
                                cv2.putText(frame, "Sleep to Sit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                            else:
                                cv2.putText(frame, "No Transition Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        # Otherwise, consider the person as stationary
                        cv2.putText(frame, "No Transition Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)
                
                x_values = [coord[0] for coord in track]
                y_values = [coord[1] for coord in track]
                combined_points = [(x, y) for x, y in zip(x_values, y_values)]

                # Draw the tracking lines
                points = np.hstack(combined_points).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()