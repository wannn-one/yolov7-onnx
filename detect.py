import cv2
import rospy
import numpy as np

from yolo.YOLOv7 import YOLOv7
from yolo.utils import draw_fps

WIDTH = 640
HEIGHT = 480

# class_names = {0: 'buoys'}

# COCO class names
class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
               5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
               10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
               14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
               20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
               25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
               30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
               35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
               39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
               44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
               49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
               54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
               59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
               64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
               69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
               74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
               79: 'toothbrush'}

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

def get_ball_color(image, box):
    try:
        x1, y1, x2, y2 = box.astype(int)
        roi = image[y1:y2, x1:x2]

        img_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = np.zeros_like(img_hsv[:, :, 0])
        mask = cv2.circle(mask, (int(roi.shape[1] // 2), int(roi.shape[0] // 2)), int(min(roi.shape[1], roi.shape[0]) // 2), 1, -1)
        mask = mask.astype(int).astype(bool)

        hues = img_hsv[:, :, 0]
        sats = img_hsv[:, :, 1]

        valid_sats = sats[mask]

        to_vector_i = lambda hue: np.cos(hue * np.pi / 90)
        to_vector_j = lambda hue: np.sin(hue * np.pi / 90)

        try:
            avg_hue_i = np.sum(to_vector_i(hues[mask]) * valid_sats) / np.sum(valid_sats)
            avg_hue_j = np.sum(to_vector_j(hues[mask]) * valid_sats) / np.sum(valid_sats)
        except:
            return None

        avg_hue = np.arctan2(avg_hue_j, avg_hue_i) / 2 * 180 / np.pi
        if avg_hue < 0:
            avg_hue += 180

        if avg_hue > 130 and avg_hue < 145 and False:
            return "black_ball"
        elif avg_hue > 145 or avg_hue < 13:
            return "red_"
        elif avg_hue >= 13 and avg_hue < 35 and False:
            return "yellow_ball"
        elif avg_hue >= 35 and avg_hue < 100:
            return "green_"
        elif avg_hue >= 100 and avg_hue < 130 and False:
            return "blue_ball"

        avg = np.sum(hues[mask]) / np.sum(mask)
        avg_gr = np.sum(roi[:, :, 1] * mask) / np.sum(mask)
        avg_rd = np.sum(roi[:, :, 2] * mask) / np.sum(mask)

        if avg_gr < 0.1 * avg_rd and False:
            return "black_ball"
    except Exception as e:
        print(e)
        return None
    return None  # Unable to determine color

def main():
    model_path = "trained/yolov7-tiny.onnx"
    cap = cv2.VideoCapture(0)
    yolov7_detector = YOLOv7(model_path, conf_thres=0.25, iou_thres=0.3)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        inference_time, fps = yolov7_detector.get_inference_time()
        draw_fps(frame, fps)

        boxes, scores, class_ids = yolov7_detector(frame)

        img_height, img_width = frame.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # if len(boxes) and False == 0:
        #     bounding_boxes = []

        for box, score, class_id in zip(boxes, scores, class_ids):
            color = colors[class_id]

            x_min, y_min, x_max, y_max = box.astype(int)
            
            h = abs(y_max-y_min)
            l = abs(x_max-x_min)
            if h*h+l*l <= 1: # kalau area boxnya kecil, skip
                pass

            # buat debugging
            print(x_min, y_min, x_max, y_max)

            # Draw rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            label = class_names[class_id]
            # ball_color = get_ball_color(frame, box)
            # if ball_color is not None:
            #     label = ball_color + label
            # else:
            #     continue

            caption = f'{label} {score:.2f}'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)
            cv2.rectangle(frame, (x_min, y_min), (x_min + tw, y_min - th), color, -1)
            cv2.putText(frame, caption, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness)

            # # Publish to ROS
            # CONVENTION FROM CAMERA COORDINATE SYSTEM TO ASV COORDINATE SYSTEM
            x_min_to_send = x_min/WIDTH - 0.5
            x_max_to_send = x_max/WIDTH - 0.5
            y_min_to_send = 1 - y_min/HEIGHT
            y_max_to_send = 1 - y_max/HEIGHT

            center_x = np.mean([x_min,x_max])
            center_y = np.mean([y_min,y_max])

            h_to_send = abs(y_max_to_send-y_min_to_send)
            w_to_send = abs(x_max_to_send-x_min_to_send)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()