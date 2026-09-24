import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load pretrained YOLOv8 model
target_class = "bottle"  # The object you want to detect

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame size: width={frame_width}, height={frame_height}")
print(f"Min coordinates: (0, 0)")
print(f"Max coordinates: ({frame_width - 1}, {frame_height - 1})")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Only proceed if the label matches your target
            if label == target_class:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                print(f"Detected {label}: ({x1}, {y1}) → ({x2}, {y2}), conf={conf:.2f}")

    cv2.imshow("YOLO Specific Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
