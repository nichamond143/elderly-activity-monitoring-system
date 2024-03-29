from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('activity-model.pt')

# Video path and capture
video_path = "videos/demo-1.mp4"
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Initialize variables
frame_count = 0
elapsed_time = 0
track_hist = defaultdict(lambda: [])
curr = None

# Dictionary to store activity log
act_log = {
    "prev": None,
    "stand": {"start_time": None, "duration": 0},
    "sit": {"start_time": None, "duration": 0},
    "sleep": {"start_time": None, "duration": 0},
    "stand_to_sit": {"start_time": None, "duration": 0},
    "sit_to_stand": {"start_time": None, "duration": 0},
    "sit_to_sleep": {"start_time": None, "duration": 0},
    "sleep_to_sit": {"start_time": None, "duration": 0},
}

# Mapping activity classes to labels
act_map = {
    0: "stand",
    1: "sleep",
    2: "sit"
}

# Main loop
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    frame_count += 1
    elapsed_time = frame_count / frame_rate

    # Push to database every 5 seconds
    if frame_count % (frame_rate * 5) == 0:
        print('PUSHED TO DATABASE...')

        # Reset activity log
        for key in act_log:
            if key != "prev":
                act_log[key]["start_time"] = None
                act_log[key]["duration"] = 0
        act_log["prev"] = None

    # Track objects in the frame
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        activity = results[0].boxes.cls.cpu().numpy().astype(int)[0]

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_hist[track_id]
            track.append((float(x), float(y), float(w), float(h)))

            # Determine current activity
            if len(track) >= 10:
                x_diff = track[-1][0] - track[-10][0]
                y_diff = track[-1][1] - track[-10][1]
                w_diff = track[-1][2] - track[-10][2]
                h_diff = track[-1][3] - track[-10][3]

                if abs(y_diff) > 10 or abs(h_diff) > 10 or abs(w_diff) > 10:
                    if h > w:
                        if h_diff < -15:
                            curr = "stand_to_sit"
                        elif h_diff > 15:
                            curr = "sit_to_stand"
                        else:
                            curr = act_map.get(activity)
                    else:
                        if w_diff > 10:
                            curr = "sit_to_sleep"
                        elif w_diff < -10:
                            curr = "sleep_to_sit"
                        else:
                            curr = act_map.get(activity)
                else:
                    curr = act_map.get(activity)
                
                # Update start time if activity just started
                if act_log[curr]["start_time"] is None:
                    act_log[curr]["start_time"] = elapsed_time

            if len(track) > 30:
                track.pop(0)

            # Draw trajectory on frame
            x_values = [coord[0] for coord in track]
            y_values = [coord[1] for coord in track]
            combined_points = [(x, y) for x, y in zip(x_values, y_values)]
            points = np.hstack(combined_points).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display information on frame
            cv2.putText(frame, f"Time: {elapsed_time:.2f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"Current Act: {curr}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            i = 0
            for key, value in act_log.items():
                if key != "prev":
                    cv2.putText(frame, f"{key}: {value['duration']:.2f} s", (10, 90 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    i += 1
    
    # Update duration if activity hasn't changed
    if act_log["prev"] is not None and act_log[curr]["start_time"] is not None and act_log["prev"] == curr:
        act_log[curr]["duration"] = elapsed_time - act_log[curr]["start_time"]

    act_log["prev"] = curr

    # Display annotated frame
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
