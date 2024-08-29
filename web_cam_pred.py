import cv2
import ultralytics
from ultralytics import YOLO


model = YOLO("E:\yolov8\hyp_AdamW_SGD_2.pt")

video = cv2.VideoCapture("E:\yolov8\TimeVideo_20240816_153301.mp4")
if not video.isOpened:
    print(f"Error loading the video file from source")
while True:
    ret , frame = video.read()

    if not ret:
        print("Error Processing the Video file , check video file format")
        break


    pred_results = model.track(frame)

    annoted_frame = pred_results[0].plot()
    cv2.imshow("Live web cam tracking window using ultralytics - YOLOv8", annoted_frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        print("You have pressed q , and Exited the live webcam winndow")
        print("Thank You for using this tool"
              )
        break
video.release()
cv2.destroyAllWindows()
