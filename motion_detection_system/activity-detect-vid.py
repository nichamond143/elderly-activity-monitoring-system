import collections
import os
import psutil
import logging
import smtplib
import time 

import cv2
import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, firestore, messaging
from ultralytics import YOLO
from screeninfo import get_monitors

# Setup logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

#NOTE: OPTIONAL PUSH TO FIREBASE FIRESTORE

# private_key = os.getenv("ELDERLY_KEY")
# doc_id = os.getenv("DOC_ID")
# email = os.getenv("APP_EMAIL")
# receiver_email = os.getenv("ADMIN_EMAIL")
# app_password = os.getenv("APP_PASSWORD")


# def initialize_firestore(private_key):
#     if not private_key:
#         logging.error("Private key not found in environment variables.")
#         return None
#     try:
#         cred = credentials.Certificate(private_key)
#         firebase_admin.initialize_app(cred)
#         return firestore.client()
#     except Exception as e:
#         logging.error(f"Error initializing Firestore: {e}")
#         return None


# def push_act_log(act_dict, db, doc_id):
#     if not db:
#         logging.error("Firestore database instance is None.")
#         return

#     try:
#         act_log = {
#             "stand": act_dict["stand"]["duration"],
#             "sit": act_dict["sit"]["duration"],
#             "sleep": act_dict["sleep"]["duration"],
#             "stand_to_sit": act_dict["stand_to_sit"]["duration"],
#             "sit_to_stand": act_dict["sit_to_stand"]["duration"],
#             "sit_to_sleep": act_dict["sit_to_sleep"]["duration"],
#             "sleep_to_sit": act_dict["sleep_to_sit"]["duration"],
#             "timestamp": firestore.SERVER_TIMESTAMP,
#         }

#         doc_ref = (
#             db.collection("patients")
#             .document(doc_id)
#             .collection("activities")
#             .document()
#         )
#         doc_ref.set(act_log)

#         # Reset activity log
#         for key in act_dict:
#             if key != "prev":
#                 act_dict[key]["start_time"] = None
#                 act_dict[key]["duration"] = 0

#         act_dict["prev"] = None
#     except Exception as e:
#         logging.error(f"Error pushing to database: {e}")

#NOTE: OPTIONAL notifiy admin by email
# def notify_admin(type):

#     # Define alert templates
#     logging.info("Notifying Admin...")
#     alerts = {
#         0: {
#             "type": "System Alert",
#             "title": "Camera Connection Issue",
#             "text": "The camera is disconnected or in use by another application. Video feed interrupted."
#         },
#         1: {
#             "type": "System Alert",
#             "title": "Camera Connection Issue",
#             "text": "Camera feed lost. Attempting to switch to backup camera."
#         },
#         2: {
#             "type": "System Warning",
#             "title": "High Resource Usage",
#             "text": "High resource usage detected. Restarting activity recognition program to avoid performance issues or system crashes."
#         },
#         3: {
#             "type": "Unknown",
#             "title": "Unknown Issue",
#             "text": "Contact Admin"
#         }
#     }

#     alert = alerts.get(type, alerts[3])
    
#     alert.update({
#         "patientID": doc_id,
#     })

#     subject = f"{alert['type']} Test"
#     message = f"{alert['title']}: {alert['text']} \nID: {alert['patientID']}"
#     text = f"Subject:{subject}\n\n{message}"

#     try:
#         logging.info("Sending email...")
#         server = smtplib.SMTP("smtp.gmail.com", 587)
#         server.starttls()
        
#         server.login(email, app_password)
#         server.sendmail(email, receiver_email, text)
#         logging.info("Email sent to Admin")
#     except Exception as e:
#         logging.error(f"Error pushing to alert: {e}")

def main():
        
    video_path = "videos/demo-1.mp4"

    # Firebase
    # db = initialize_firestore(private_key)
    # if db is None:
    #     logging.error("Exiting due to Firestore initialization failure.")
    #     return

    # Initialize YOLO Model
    try:
        model = YOLO("yolo-Weights/activity-model-v8m.pt")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return

    monitors = get_monitors()
    if not monitors:
        logging.error("No monitors found.")
        return

    screen_width, screen_height = monitors[0].width, monitors[0].height
    frame_rate, frame_count, elapsed_time = cap.get(cv2.CAP_PROP_FPS), 0, 0
    
    #TODO: Set variables
    frequency, CPU_THRESH, DUR_THRESH = 5, 80.0, 600.0

    track_hist = collections.defaultdict(list)
    curr, cpu, high_usage_start, high_usage_dur = None, None, None, None


    # Activity classes map
    act_map = {0: "stand", 1: "sleep", 2: "sit"}
    MAX_TRACK_HISTORY = 30

    # Activity log dictionary
    act_dict = {
        "prev": None,
        "stand": {"start_time": None, "duration": 0},
        "sit": {"start_time": None, "duration": 0},
        "sleep": {"start_time": None, "duration": 0},
        "stand_to_sit": {"start_time": None, "duration": 0},
        "sit_to_stand": {"start_time": None, "duration": 0},
        "sit_to_sleep": {"start_time": None, "duration": 0},
        "sleep_to_sit": {"start_time": None, "duration": 0},
    }

    while cap.isOpened():

        cap.set(3, screen_width)
        cap.set(4, screen_height)

        success, frame = cap.read()

        if not success:
            logging.info("End of video stream.")
            break

        frame = cv2.resize(frame, (screen_width, screen_height))
        frame_count += 1
        elapsed_time = frame_count / frame_rate

        # Display information on frame
        cv2.putText(
            frame,
            f"Time: {elapsed_time:.2f} s",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (255, 255, 255),
            2,
        )

        # Track object in frame
        try:
            results = model.track(frame, persist=True)
        except Exception as e:
            logging.error(f"Error tracking objects: {e}")
            continue

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
                            if h_diff < -15 and y_diff > 10:
                                curr = "stand_to_sit"
                            elif h_diff > 15 and y_diff < -10:
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
                    if act_dict[curr]["start_time"] is None:
                        act_dict[curr]["start_time"] = round(elapsed_time, 2)

                    # Pushed to database every frequency seconds
                    # if frame_count % (frame_rate * frequency) == 0:
                    #     logging.info("PUSHED TO DATABASE...")
                    #     push_to_database(act_dict, db, doc_id)

                    cv2.putText(
                        frame,
                        f"Current Act: {curr}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.25,
                        (255, 255, 255),
                        2,
                    )

                    i = 0
                    for key, value in act_dict.items():
                        if key != "prev":
                            cv2.putText(
                                frame,
                                f"{key}: {value['duration']} s",
                                (10, 120 + (i * 35)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 0, 0),
                                2,
                            )
                            i += 1

                    cv2.putText(
                        frame,
                        f"x_diff: {x_diff:.2f}",
                        (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"y_diff: {y_diff:.2f}",
                        (10, 400),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 0),
                        2,
                    )

                else:
                    cv2.putText(
                        frame,
                        f"Calibrating...",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.25,
                        (255, 255, 255),
                        2,
                    )

                if len(track) > MAX_TRACK_HISTORY:
                    track.pop(0)

        # Update duration if activity hasn't changed
        if (
            act_dict["prev"] is not None
            and act_dict[curr]["start_time"] is not None
            and act_dict["prev"] == curr
        ):
            act_dict[curr]["duration"] = round(
                elapsed_time - act_dict[curr]["start_time"], 2
            )

        act_dict["prev"] = curr

        #CPU Monitoring
        cpu = psutil.cpu_percent()
        logging.info(f"cpu percentage: {cpu}")
        if cpu > CPU_THRESH:
            if high_usage_start is None:
                high_usage_start = time.time()
            
            high_usage_dur = time.time() - high_usage_start
            
            if high_usage_dur > DUR_THRESH:
                logging.critical("High memory usage for prolonged duration detected!")
                # notify_admin(type=2)
                break

        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
