import cv2 as cv
from ultralytics import YOLO

def identify_labels_webcam(device, model, labels, border_colours):
    # Set up webcam
    vid = cv.VideoCapture(device, cv.CAP_DSHOW)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while vid.isOpened():
        current_boxes = []
        labels_detected = []

        ret, frame = vid.read()

        if not ret or frame is None:
            # Empty frame so skip loop
            continue

        results = model(frame, verbose=False)
        detections = results[0].boxes # Extract bounding boxes

        # Go through each detection and get bbox coords, confidence, and class
        for i in range(len(detections)):
            # Get bounding box coordinates
            # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
            xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int
            current_boxes.append([xmin, xmax, ymin, ymax])

            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item()) # Integer which identifies what type of object
            classname = labels[classidx]

            # Get bounding box confidence
            conf = detections[i].conf.item()

            # Draw box if confidence threshold is higher than 50%%
            if conf > 0.5:
                color = border_colours[classidx % 10]
                cv.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

                label = f'{classname}: {int(conf*100)}%'
                labels_detected.append(classname)

                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv.FILLED) # Draw white box to put label text in
                cv.putText(frame, label, (xmin, label_ymin-7), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

        # Add text to webcam
        cv.putText(frame, "________________________________", (25, 43), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 40)
        cv.putText(frame, "Press q when all labels are detected", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the captured frame
        cv.imshow('Live webcam', frame)

        # Press q esc to escape
        if cv.waitKey(20) & 0xff == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()

    return labels_detected

if __name__ == "__main__":
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

    # YOLO model
    model = YOLO("my_model.pt")
    labels = model.names # int identifier: 'label'

    labels_detected = identify_labels_webcam(0, model, labels, border_colours)
    print(labels_detected)
    #print(identify_labels(vid, model, labels, border_colours))