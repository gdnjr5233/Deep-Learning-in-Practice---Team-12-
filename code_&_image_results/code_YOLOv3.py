import cv2
import numpy as np

# Load the YOLOv3 model
net = cv2.dnn.readNet(r"C:\Users\gdnjr5233_YOLO\Desktop\YOLOv3\cfg\yolov3-spp.cfg", r"C:\Users\gdnjr5233_YOLO\Desktop\YOLOv3\weights\yolov3-spp.weights")

# Get the names of the output layers
layer_names = net.getUnconnectedOutLayersNames()

# Load the class label file
with open(r"C:\Users\gdnjr5233_YOLO\Desktop\YOLOv3\labels_txt\traffic_sign_coco_names.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the image
image = cv2.imread(r"C:\Users\gdnjr5233_YOLO\Desktop\YOLOv3\test_images\test_6.jpg")

# Get the height and width of the image
height, width, _ = image.shape

# Create a blob (binary large object) to extract information from the image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the blob as input to the model
net.setInput(blob)

# Run forward pass
outputs = net.forward(layer_names)

# Initialize lists to store box coordinates, confidences, and class IDs
bboxes = []
confidences = []
class_ids = []

# Iterate through the output layers and extract detection results
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            # Coordinates of the detection box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Coordinates of the top-left corner of the detection box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Save box information
            bboxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-Maximum Suppression
confidence_threshold = 0.5  # Confidence threshold
nms_threshold = 0.4  # NMS threshold
indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)

# Iterate through the retained box indices and draw detection boxes on the image
for i in indices.flatten():
    x, y, w, h = bboxes[i]
    class_id = class_ids[i]

    # Skip labels you do not want to display
    # if classes[class_id] in ['speed_limit_30', 'speed_limit_40', 'speed_limit_50', 'traffic light']:
    #     continue

    # Draw detection box on the image
    color = (0, 255, 0)  # Green
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

    # Set font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Increase font size
    font_thickness = 1  # Increase font thickness

    # Set text color
    text_color = (0, 0, 255)  # Red

    # Display class name and confidence on the detection box
    text = f"{classes[class_id]}: {confidences[i]:.2f}"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = x
    text_y = y - 10  # Move the text upward to avoid tight connection with the detection box

    # Draw text outline
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness + 1, cv2.LINE_AA)

    # Draw text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Display the result
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
