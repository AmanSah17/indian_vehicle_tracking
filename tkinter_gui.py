import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tkinter import Tk, filedialog

def select_file(file_type, title):
    # Open a file dialog to select the file
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title=title, 
        filetypes=file_type
    )
    return file_path

def detect_objects(model_path, image_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Perform object detection
    results = model(image)

    # Plot the results
    plt.imshow(results[0].plot())
    plt.title("YOLOv8 Object Detection")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Select the model file
    model_path = select_file(
        file_type=[('YOLO Model Files', '*.pt')],
        title="Select YOLOv8 Model File"
    )

    if not model_path:
        print("Model file not selected. Exiting.")
        exit()

    # Select the image file
    image_path = select_file(
        file_type=[('Image Files', '*.jpg;*.jpeg;*.png;*.bmp;*.tiff')],
        title="Select Image File"
    )

    if not image_path:
        print("Image file not selected. Exiting.")
        exit()

    # Run detection
    detect_objects(model_path, image_path)
