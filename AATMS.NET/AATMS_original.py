import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import pandas as pd
import os

# Global variables for drawing the lines
lines = []
drawing = False
current_line = []
unique_id_counter = 0

# Pseudocode for blinking effect
blink_duration = 10  # frames
blink_state = {}  # {track_id: {line_idx: frames_left}}

def draw_line(event, x, y, flags, param):
    global drawing, current_line
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_line = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = param.copy()
        cv2.line(img_copy, current_line[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Line Drawing", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_line.append((x, y))
        lines.append(current_line)

def get_lines(frame, max_lines=4):
    global lines, drawing, current_line
    lines = []
    drawing = False
    current_line = []
    cv2.namedWindow("Line Drawing")
    cv2.setMouseCallback("Line Drawing", draw_line, frame)

    while len(lines) < max_lines:
        img_copy = frame.copy()
        for idx, line in enumerate(lines):
            cv2.line(img_copy, line[0], line[1], (10, 255, 100), 2)
            cv2.putText(img_copy, f'L{idx+1}', (line[0][0], line[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Line Drawing", img_copy)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyWindow("Line Drawing")
    return lines, img_copy

def is_crossed(bottom_center, line, threshold=5):
    line_y1 = line[0][1]
    line_y2 = line[1][1]
    line_y_min = min(line_y1, line_y2) - threshold
    line_y_max = max(line_y1, line_y2) + threshold
    return line_y_min <= bottom_center[1] <= line_y_max

def track_objects_in_lines(video_path, model_path, lines, frame_skip, recording_start_time):
    global unique_id_counter
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    tracking_info = {}
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            results = model.track(frame, persist=True, conf=0.2)
            elapsed_time = timedelta(seconds=frame_count / fps)
            current_time = recording_start_time + elapsed_time

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls) if box.cls is not None else None
                        if class_id is None or class_id >= len(model.names):
                            continue
                        
                        track_id = int(box.id) if box.id is not None else None
                        if track_id is None:
                            continue
                        
                        class_name = model.names[class_id] if class_id is not None else "unknown"
                        
                        if track_id not in tracking_info:
                            tracking_info[track_id] = {
                                'class': class_name,
                                'lines': {i: {'crossed': False, 'timestamp': 'NIL'} for i in range(len(lines))},
                                'latitude': None,
                                'longitude': None
                            }

                        bbox = box.xyxy.cpu().numpy().flatten()
                        bottom_center = ((bbox[0] + bbox[2]) // 2, bbox[3])

                        for i, line in enumerate(lines):
                            if is_crossed(bottom_center, line):
                                if not tracking_info[track_id]['lines'][i]['crossed']:
                                    tracking_info[track_id]['lines'][i]['crossed'] = True
                                    tracking_info[track_id]['lines'][i]['timestamp'] = current_time.strftime("%Y-%m-%d %H:%M:%S")

            annotated_frame = results[0].plot()
            cv2.imshow("AATM(s) Detection & Tracking Window ", annotated_frame)

            frame_with_lines = frame.copy()
            for idx, line in enumerate(lines):
                cv2.line(frame_with_lines, line[0], line[1], (100, 255, 10), 2)
                cv2.putText(frame_with_lines, f'L{idx+1}', (line[0][0], line[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 255, 100), 2)
            
            cv2.imshow("Line Drawing for Defined ROI(s)", frame_with_lines)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    data = []
    for track_id, info in tracking_info.items():
        row = {
            'Vehicle ID': f"{info['class']}_{track_id}",
            'Class': info['class'],
            'Latitude': info['latitude'],
            'Longitude': info['longitude']
        }
        for i in range(len(lines)):
            row[f'Line {i+1}'] = 1 if info['lines'][i]['crossed'] else 0
            row[f'Timestamp Line {i+1}'] = info['lines'][i]['timestamp']
        data.append(row)

    df = pd.DataFrame(data)

    for i in range(len(lines)):
        df[[f'Year Line {i+1}', f'Month Line {i+1}', f'Day Line {i+1}', 
            f'Hour Line {i+1}', f'Minute Line {i+1}', f'Second Line {i+1}']] = \
            df[f'Timestamp Line {i+1}'].apply(
                lambda ts: pd.Series(pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S").strftime('%Y %m %d %H %M %S').split()) 
                if ts != 'NIL' else pd.Series([None]*6)
            )

    def determine_in_out(row):
        if row['Line 1'] == 1 and row['Line 2'] == 1:
            return 'IN/OUT'
        elif row['Line 1'] == 1:
            return 'IN'
        elif row['Line 2'] == 1:
            return 'OUT'
        return 'UNKNOWN'

    df['IN/OUT'] = df.apply(determine_in_out, axis=1)
    return df

def save_to_csv(data, save_path):
    data.to_csv(save_path, index=False)

def main(video_path, model_path, max_lines, frame_skip, save_directory):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    assert ret, "Error reading the first frame of the video"
    cap.release()

    print("Draw the lines of interest (ROIs). Press 'q' when done.")
    lines, img_with_lines = get_lines(frame, max_lines=max_lines)
    print(f"{len(lines)} lines drawn.")

    frame_skip_value = 3 if frame_skip else 1
    recording_start_time = datetime.now()

    data = track_objects_in_lines(video_path, model_path, lines, frame_skip_value, recording_start_time)

    file_prefix = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(save_directory, f"{file_prefix}_counting.csv")
    save_to_csv(data, save_path)

    img_save_path = os.path.join(save_directory, f"{file_prefix}_lines.png")
    cv2.imwrite(img_save_path, img_with_lines)

if __name__ == "__main__":
    main() 