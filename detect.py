import cv2

from yolov7 import YOLOv7
from yolov7.utils import draw_fps

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv7 object detector
model_path = "models/only_ball.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.8, iou_thres=0.5)

while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    inference_time, fps = yolov7_detector.get_inference_time()
    draw_fps(frame, fps)

    # Update object localizer
    boxes, scores, class_ids = yolov7_detector(frame)

    combined_img = yolov7_detector.draw_detections(frame)
    cv2.imshow("Webcam", combined_img)
    

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
