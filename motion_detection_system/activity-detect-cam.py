import collections
import os
import time

import cv2
import firebase_admin
import numpy as np
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
from ultralytics import YOLO

load_dotenv()

private_key = os.getenv('ELDERLY_KEY')
doc_id = os.getenv('DOC_ID')

def initialize_firestore(private_key):
    try:
        cred = credentials.Certificate(private_key)
        firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        print(f"Error initializing Firestore: {e}")
        return None

def push_to_database(act_dict, db, doc_id):
    if not db or not doc_id:
        print("Database or Document ID is not initialized.")
        return
    
    act_log = {
        'stand': act_dict["stand"]["duration"],
        'sit': act_dict["sit"]["duration"],
        'sleep': act_dict["sleep"]["duration"],
        'stand_to_sit': act_dict["stand_to_sit"]["duration"],
        'sit_to_stand': act_dict["sit_to_stand"]["duration"],
        'sit_to_sleep': act_dict["sit_to_sleep"]["duration"],
        'sleep_to_sit': act_dict["sleep_to_sit"]["duration"],
        'timestamp': firestore.SERVER_TIMESTAMP
    }

    try:
        doc_ref = db.collection('patients').document(doc_id).collection('activities').document()
        doc_ref.set(act_log)
    except Exception as e:
        print(f"Error pushing to database: {e}")
        return

    # Reset activity log
    for key in act_dict:
        if key != "prev":
            act_dict[key]["start_time"] = 0
            act_dict[key]["duration"] = 0

    act_dict["prev"] = None

def main():
    db = initialize_firestore(private_key)
    if not db:
        print("Failed to initialize Firestore. Exiting.")
        return

    # Initialize YOLO Model
    try:
        model = YOLO('activity-model.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    cap = None
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open video capture.")
            return

        cap.set(3, 720)  # width
        cap.set(4, 480)  # height

        frequency = 10
        track_hist = collections.defaultdict(lambda: [])
        curr = None
        start_time = time.time()
        last_push_time = time.time()

        # Activity log dictionary
        act_dict = {
            "prev": None,
            "stand": {"start_time": 0, "duration": 0},
            "sit": {"start_time": 0, "duration": 0},
            "sleep": {"start_time": 0, "duration": 0},
            "stand_to_sit": {"start_time": 0, "duration": 0},
            "sit_to_stand": {"start_time": 0, "duration": 0},
            "sit_to_sleep": {"start_time": 0, "duration": 0},
            "sleep_to_sit": {"start_time": 0, "duration": 0},
        }

        # Activity classes map
        act_map = {
            0: "stand",
            1: "sleep",
            2: "sit"
        }

        MAX_TRACK_HISTORY = 1000  # or some other suitable value

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            elapsed_time = time.time() - start_time

            # Display information on frame
            cv2.putText(frame, f"Time: {elapsed_time:.2f} s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2)
            cv2.putText(frame, f"Current Act: {curr}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2)
            i = 0
            for key, value in act_dict.items():
                if key != "prev":
                    cv2.putText(frame, f"{key}: {value['duration']} s", (10, 120 + (i * 35)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                    i += 1

            # Push to DB every frequency seconds
            if time.time() - last_push_time >= frequency:
                print('PUSHED TO DATABASE...')
                push_to_database(act_dict, db, doc_id)
                last_push_time = time.time()

            # Track object in frame
            results = model.track(frame, persist=True)
            if results and results[0].boxes.id is not None:
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
                                if h_diff < -15 and y_diff < 5:
                                    curr = "stand_to_sit"
                                elif h_diff > 15 and y_diff > 5:
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
                        if act_dict[curr]["start_time"] == 0:
                            act_dict[curr]["start_time"] = round(elapsed_time, 2)

                    if len(track) > 30:
                        track.pop(0)

                    # Draw trajectory on frame
                    x_values = [coord[0] for coord in track]
                    y_values = [coord[1] for coord in track]
                    combined_points = [(x, y) for x, y in zip(x_values, y_values)]
                    points = np.array(combined_points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                # Clean up track history
                if len(track_hist) > MAX_TRACK_HISTORY:
                    for track_id in list(track_hist.keys())[:len(track_hist) - MAX_TRACK_HISTORY]:
                        del track_hist[track_id]

            # Update duration if activity hasn't changed
            if act_dict["prev"] is not None and act_dict[curr]["start_time"] != 0 and act_dict["prev"] == curr:
                act_dict[curr]["duration"] = round(elapsed_time - act_dict[curr]["start_time"], 2)

            act_dict["prev"] = curr

            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Error during video processing: {e}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
