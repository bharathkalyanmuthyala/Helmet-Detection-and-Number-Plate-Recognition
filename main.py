from ultralytics import YOLO
import math
import cv2
import cvzone
import torch
from image_to_text import predict_number_plate
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from paddleocr import PaddleOCR

cap = cv2.VideoCapture("videos/22.mp4")  # For videos

#model = YOLO(r"D:\minor333\Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates\runs\detect\train2\weights\best.pt")
model=YOLO('./best.pt')
# Dynamically set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classNames = ["with helmet", "without helmet", "rider", "number plate"]
num = 0
old_npconf = 0

# Grab the width, height, and fps of the frames in the video stream
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load OCR model into memory

while True:
    success, img = cap.read()
    if not success:  # Check if the frame was read successfully
        break

    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device=device)

    for r in results:
        boxes = r.boxes
        li = {}
        rider_box = []
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), dim=1)

        try:
            new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]

            # Get rows where the class is 'rider' (class index 2)
            indices = torch.where(new_boxes[:, -1] == 2)
            rows = new_boxes[indices]

            for box in rows:
                x1, y1, x2, y2 = map(int, box[:4])
                rider_box.append((x1, y1, x2, y2))
        except Exception as e:
            print(f"Error processing bounding boxes: {e}")

        for i, box in enumerate(new_boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box[4] * 100)) / 100
            cls = int(box[5])

            if (classNames[cls] == "without helmet" and conf >= 0.5) or \
               (classNames[cls] == "rider" and conf >= 0.45) or \
               (classNames[cls] == "number plate" and conf >= 0.5):

                if classNames[cls] == "rider":
                    rider_box.append((x1, y1, x2, y2))

                if rider_box:
                    for j, rider in enumerate(rider_box):
                        if x1 + 10 >= rider[0] and y1 + 10 >= rider[1] and x2 <= rider[2] and y2 <= rider[3]:
                            cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                            cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                               offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                            li.setdefault(f"rider{j}", [])
                            li[f"rider{j}"].append(classNames[cls])

                            if classNames[cls] == "number plate":
                                crop = img[y1:y1 + h, x1:x1 + w]

                        if li and f"rider{j}" in li:
                            if len(list(set(li[f"rider{j}"]))) == 3:
                                try:
                                    vechicle_number, conf = predict_number_plate(crop, ocr)
                                    if vechicle_number and conf:
                                        cvzone.putTextRect(img, f"{vechicle_number} {round(conf * 100, 2)}%",
                                                           (x1, y1 - 50), scale=1.5, offset=10, thickness=2,
                                                           colorT=(39, 40, 41), colorR=(105, 255, 255))
                                except Exception as e:
                                    print(f"OCR Error: {e}")

    output.write(img)
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
 