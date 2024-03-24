from ultralytics import YOLO
import cv2

def main():

    prev = None

    model = YOLO('activity-model.pt')

    video_path = "videos/demo-1.mp4"

    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():

        success, frame = cap.read()

        if success:

            results = model.track(frame, persist=True)

            print(results[0].boxes.id)
            if results[0].boxes.id != None:
                track_id = results[0].boxes.id[0].item()
                activity = results[0].boxes.cls.cpu().numpy().astype(int)[0]
                conf = results[0].boxes.conf.cpu().numpy().astype(float)[0]
                xywh = results[0].boxes.xywh.cpu().numpy()[0]

                width = xywh[2]
                height = xywh[3]

                # Set current 
                curr = (width, height)

                # Detect transition
                if prev is not None:
                    curr_width, curr_height = curr
                    prev_width, prev_height = prev

                    # Calculate height and width differences
                    height_diff = curr_height - prev_height
                    width_diff = curr_width - prev_width
                    cv2.putText(frame, f"Height: {height:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    cv2.putText(frame, f"Width: {width:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    cv2.putText(frame, f"Height Diff: {height_diff:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    cv2.putText(frame, f"Width Diff: {width_diff:.3f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    # Transition Logic
                    if -1 < height_diff < 1 and -1 < width_diff < 1:
                        cv2.putText(frame, "No Transition Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        if height > width: 
                            if height_diff < 0 or height_diff < width_diff:
                                cv2.putText(frame, "Stand to Sit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            else:
                                cv2.putText(frame, "Sit to Stand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            if width_diff > 0 or height_diff < width_diff:
                                cv2.putText(frame, "Sit to Sleep", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            else:
                                cv2.putText(frame, "Sleep to Sit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)

                # Update prev
                prev = curr

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


