import cv2
import numpy as np

def load_yolo_model(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNet(cfg_path, weights_path)
    with open(names_path) as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

def get_output_layers(net):
    layer_names = net.getUnconnectedOutLayersNames()
    return layer_names

def postprocess(image, outputs, classes, confidence_threshold=0.5, nms_threshold=0.4):
    height, width, _ = image.shape
    bboxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
    for i in indices.flatten():
        x, y, w, h = bboxes[i]
        class_id = class_ids[i]
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 0, 255)
        text = f"{classes[class_id]}: {confidences[i]:.2f}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x
        text_y = y - 10
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness + 1, cv2.LINE_AA)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

def main():
    cfg_path = "cfg/yolov3-spp.cfg", "cfg\yolov4.cfg"
    weights_path = "weights/yolov3-spp.weights", "weights\yolov4.weights"
    names_path = "labels_txt/traffic_sign_coco_names.txt"

    net, classes = load_yolo_model(cfg_path, weights_path, names_path)

    image_path = "test_images/test_6.jpg"
    image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward(get_output_layers(net))

    postprocess(image, outputs, classes)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()