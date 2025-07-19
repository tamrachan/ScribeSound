from identify_labels import identify_labels_webcam
from instrument_gui import CustomKeys
from play_sounds import detect_sounds

from PyQt5.QtWidgets import QApplication
import sys
from ultralytics import YOLO
import mediapipe.python.solutions.hands as mp_hands

# Set bounding box colors (using the Tableu 10 color scheme)
border_colours = [
    (255, 179, 186),  # Pastel Pink
    (255, 223, 186),  # Pastel Orange
    (255, 255, 186),  # Pastel Yellow
    (186, 255, 201),  # Pastel Green
    (186, 225, 255),  # Pastel Blue
    (210, 190, 255),  # Pastel Purple
    (202, 255, 229),  # Pastel Mint
    (240, 200, 255),  # Pastel Lavender
    (255, 204, 204),  # Pastel Peach
    (204, 255, 255)   # Pastel Cyan
]

WEBCAM_DEVICE = 0

# YOLO model
try:
    model = YOLO("my_model.pt")
    labels = model.names # int identifier: 'label'
except Exception as e:
    print("Failed to load model:", e)
    sys.exit(1)

# 1. Detect which labels to map sounds to
labels_detected = identify_labels_webcam(WEBCAM_DEVICE, model, labels, border_colours)
print("labels_detected: ", labels_detected)

# 2. Choose the sounds to map the labels to
app = QApplication(sys.argv)
UIWindow = CustomKeys(labels_detected)
app.exec_()

# Initialise the Hands model
hands = mp_hands.Hands(
    static_image_mode = False,  # Set to False for processing video/webcam frames
    max_num_hands = 2,           # Maximum number of hands to detect
    min_detection_confidence = 0.5  # Confidence threshold must be above 50% to display hand detection
)

try:
    print("retrieved data: ", UIWindow.data)
except:
    print("No data fed")
    exit(1)

# 3. Have fun with the webcam AI and sounds!
detect_sounds(WEBCAM_DEVICE, model, labels, hands, UIWindow.data)