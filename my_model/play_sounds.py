import cv2 as cv
from ultralytics import YOLO
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import pygame
import os

def detect_sounds(device, model, labels, hands, data):
    vid = cv.VideoCapture(device, cv.CAP_DSHOW)

    vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    freeze_frame = False

    while vid.isOpened():
        ret, frame = vid.read()

        if not freeze_frame:
            current_boxes = []
        else:
            frame = cv.flip(frame, 1)
        
        if not ret or frame is None:
            # print("Empty frame received, skipping...")
            continue

        height, width = frame.shape[:2]
        
        # Convert the frame from BGR to RGB (required by MediaPipe)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process the frame for hand detection and tracking
        hands_detected = hands.process(frame)

        # Convert the frame back from RGB to BGR (required by OpenCV)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        results = model(frame, verbose=False)
        detections = results[0].boxes # Extract bounding boxes

        # Go through each detection and get bbox coords, confidence, and class
        for i in range(len(detections)):
            # Get bounding box coordinates
            # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
            xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int
            
            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item()) # Integer which identifies what type of object
            classname = labels[classidx]
            # Get bounding box confidence
            conf = detections[i].conf.item()

            if not freeze_frame:
                current_boxes.append([xmin, xmax, ymin, ymax, classname, conf])

                if conf > 0.5:
                    draw_detection_box(frame, current_boxes[i])
           

            # Draw box if confidence threshold is higher than 50%%
            
                # color = border_colours[classidx % 10]
                # cv.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

                # label = f'{classname}: {int(conf*100)}%'
                # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
                # label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                # cv.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv.FILLED) # Draw white box to put label text in
                # cv.putText(frame, label, (xmin, label_ymin-7), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text
        
        # If hands are detected, draw landmarks and connections on the frame
        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

                #print(current_boxes)
                #print(int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height))
                for box in current_boxes:
                    draw_detection_box(frame, box)

                    thumb_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
                    thumb_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
                    index_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                    index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                    middle_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
                    middle_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)
                    ring_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
                    ring_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)
                    pinky_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
                    pinky_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)

                    if box[0] <= thumb_x <= box[1] and box[2] <= thumb_y <= box[3]:
                        play_note(data, box[4])
                    if box[0] <= index_x <= box[1] and box[2] <= index_y <= box[3]:
                        play_note(data, box[4])
                    if box[0] <= middle_x <= box[1] and box[2] <= middle_y <= box[3]:
                        play_note(data, box[4])
                    if box[0] <= ring_x <= box[1] and box[2] <= ring_y <= box[3]:
                        play_note(data, box[4])
                    if box[0] <= pinky_x <= box[1] and box[2] <= pinky_y <= box[3]:
                        play_note(data, box[4])
                
        # Add text to webcam
        if freeze_frame:
            cv.putText(frame, "___________________________________________", (25, 43), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 223, 186), 40)
            cv.putText(frame, "FRAME FROZEN - press f to unfreeze the frame", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            cv.putText(frame, "________________________", (25, 43), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 40)
            cv.putText(frame, "Press f to freeze the boxes", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(frame, "_____________", (25, 88), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 40)
        cv.putText(frame, "Press q to exit", (20, 95), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the captured frame
        cv.imshow('Live webcam', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('f'):
            if freeze_frame:
                freeze_frame = False
            else:
                freeze_frame = True
        if key == ord('q'):
            # Press q esc to escape
            break
        

    vid.release()

def draw_detection_box(frame, current_box):
    xmin, xmax, ymin, ymax, classname, conf = current_box
    # box_colour = (186, 225, 255) # is Pastel Orange
    box_colour = (255, 223, 186)

    cv.rectangle(frame, (xmin,ymin), (xmax,ymax), box_colour, 2)

    label = f'{classname}: {int(conf*100)}%'
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
    cv.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), box_colour, cv.FILLED) # Draw white box to put label text in
    cv.putText(frame, label, (xmin, label_ymin-7), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

def play_note(data, label):
    print(data)
    print(label)

    pygame.mixer.init()

    for entry in data:
        if entry['label'] == label:
            sound_path = os.path.join("notes", entry['instrument'], entry['note']+".wav")
            pygame.mixer.Sound(sound_path).play()

# For testing the file
if __name__ == "__main__":
    # Initialise the Hands model
    hands = mp_hands.Hands(
        static_image_mode=False,  # Set to False for processing video frames
        max_num_hands=2,           # Maximum number of hands to detect
        min_detection_confidence=0.5  # Minimum confidence threshold for hand detection
    )

    # YOLO model
    model = YOLO("my_model.pt")
    labels = model.names # int identifier: 'label'

    data = [{'label': '5', 'instrument': 'Keyboard', 'note': 'A5'}, {'label': '4', 'instrument': 'Keyboard', 'note': 'A7'}, {'label': '1', 'instrument': 'Keyboard', 'note': 'B0'}, {'label': '6', 'instrument': 'Keyboard', 'note': 'B1'}, {'label': '0', 'instrument': 'Keyboard', 'note': 'C3'}, {'label': '7', 'instrument': 'Keyboard', 'note': 'C1'}, {'label': '8', 'instrument': 'Keyboard', 'note': 'C2'}, {'label': '3', 'instrument': 'Keyboard', 'note': 'D1'}, {'label': '2', 'instrument': 'Keyboard', 'note': 'D2'}, {'label': '9', 'instrument': 'Keyboard', 'note': 'E2'}]

    detect_sounds(0, model, labels, hands, data)