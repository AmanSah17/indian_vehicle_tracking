import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
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

def select_file(title, filetypes):
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def select_video_file():
    return select_file("Select a Video File", [("MP4 files", "*.mp4"), ("All files", "*.*")])

def select_model_file():
    return select_file("Select a YOLO Model File", [("PT files", "*.pt"), ("All files", "*.*")])

def select_directory():
    return filedialog.askdirectory(title="Select Directory to Save Processed Data")

def is_crossed(bottom_center, line, threshold=5):
    line_y1 = line[0][1]
    line_y2 = line[1][1]
    line_y_min = min(line_y1, line_y2) - threshold
    line_y_max = max(line_y1, line_y2) + threshold

    # Check if the bottom center y-coordinate is within the threshold range of the line's y-coordinates
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

        # Skip frames for faster processing
        if frame_count % frame_skip == 0:
            results = model.track(frame, persist=True, conf=0.2)
            elapsed_time = timedelta(seconds=frame_count / fps)
            current_time = recording_start_time + elapsed_time

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if the class ID is valid
                        class_id = int(box.cls) if box.cls is not None else None
                        if class_id is None or class_id >= len(model.names):
                            continue  # Skip unknown classes
                        
                        # Check if the track ID is valid
                        track_id = int(box.id) if box.id is not None else None
                        if track_id is None:
                            continue  # Skip unknown IDs
                        
                        class_name = model.names[class_id] if class_id is not None else "unknown"
                        
                        # Initialize tracking info for new objects
                        if track_id not in tracking_info:
                            tracking_info[track_id] = {
                                'class': class_name,
                                'lines': {i: {'crossed': False, 'timestamp': 'NIL'} for i in range(len(lines))},
                                'latitude': None,
                                'longitude': None
                            }

                        # Calculate bottom center of the bounding box using the result.bbox
                        bbox = box.xyxy.cpu().numpy().flatten()
                        bottom_center = ((bbox[0] + bbox[2]) // 2, bbox[3])

                        # Your existing logic to check line crossings
                        for i, line in enumerate(lines):
                            if is_crossed(bottom_center, line):
                                if not tracking_info[track_id]['lines'][i]['crossed']:
                                    tracking_info[track_id]['lines'][i]['crossed'] = True
                                    tracking_info[track_id]['lines'][i]['timestamp'] = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Visualization and display code here

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("AATM(s) Detection & Tracking Window ", annotated_frame)

            # Display the lines and labels
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

    # Prepare data for CSV
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

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Add separate date and time columns
    for i in range(len(lines)):
        df[[f'Year Line {i+1}', f'Month Line {i+1}', f'Day Line {i+1}', f'Hour Line {i+1}', f'Minute Line {i+1}', f'Second Line {i+1}']] = df[f'Timestamp Line {i+1}'].apply(lambda ts: pd.Series(pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S").strftime('%Y %m %d %H %M %S').split()) if ts != 'NIL' else pd.Series([None]*6))

    # Add IN/OUT column
    def determine_in_out(row):
        if row['Line 1'] == 1 and row['Line 2'] == 1:
            return 'IN/OUT'
        elif row['Line 1'] == 1:
            return 'IN'
        elif row['Line 2'] == 1:
            return 'OUT'
        return 'UNKNOWN'

    df['IN/OUT'] = df.apply(determine_in_out, axis=1)

    # Save the modified DataFrame to CSV
    return df

def save_to_csv(data, save_path):
    data.to_csv(save_path, index=False)

def main():
    root = tk.Tk()
    root.withdraw()

    video_path = select_video_file()
    if not video_path:
        messagebox.showerror("Error", "No video file selected!")
        return

    model_path = select_model_file()
    if not model_path:
        messagebox.showerror("Error", "No model file selected!")
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    assert ret, "Error reading the first frame of the video"
    cap.release()

    max_lines = int(input("Enter the maximum number of lines to draw: "))
    print("Draw the lines of interest (ROIs). Press 'q' when done.")
    lines, img_with_lines = get_lines(frame, max_lines=max_lines)
    print(f"{len(lines)} lines drawn.")

    save_directory = select_directory()
    if not save_directory:
        messagebox.showerror("Error", "No directory selected to save data!")
        return

    frame_skip_input = messagebox.askyesno("Frame Skip", "Do you want to skip frames for faster processing?")
    frame_skip = 3 if frame_skip_input else 1

    timestamp_format = "%Y-%m-%d %H:%M:%S"
    recording_start_time = datetime.now()

    data = track_objects_in_lines(video_path, model_path, lines, frame_skip, recording_start_time)

    file_prefix = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(save_directory, f"{file_prefix}_counting.csv")

    save_to_csv(data, save_path)
    messagebox.showinfo("Success", f"Counting data saved to {save_path}")

    # Save the image with drawn lines
    img_save_path = os.path.join(save_directory, f"{file_prefix}_lines.png")
    cv2.imwrite(img_save_path, img_with_lines)
    messagebox.showinfo("Success", f"First frame with lines saved to {img_save_path}")

if __name__ == "__main__":
    main()