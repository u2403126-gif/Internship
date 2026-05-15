from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# COCO class ID for cell phone
PHONE_CLASS_ID = 67

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Get detections
    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])

            # Detect only phones
            if cls_id == PHONE_CLASS_ID:

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Confidence score
                conf = float(box.conf[0])

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label
                label = f"Phone {conf:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # Show output
    cv2.imshow("Phone Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()