import cv2
import os
import sys
import numpy as np
import tensorflow as tf
import pyautogui
import collections
import threading, queue

# -------------------------
# 1) Load Model and Labels
# -------------------------
model = tf.keras.models.load_model('bestModel.keras')
print("Model loaded successfully.")

class_labels = ["doubleClick", "drag", "drop", "leftClick", "moveCursor", "rightClick"]

# -------------------------
# Gesture Icons + Descriptions
# -------------------------
gesture_icons_info = {
    "doubleClick": ("icon_images/doubleClick.png", "Double Click"),
    "drag":        ("icon_images/drag.png",        "Click and hold (Drag)"),
    "drop":        ("icon_images/drop.png",        "Release mouse (Drop)"),
    "leftClick":   ("icon_images/leftClick.png",   "Left Mouse Click"),
    "moveCursor":  ("icon_images/moveCursor.png",  "Move Mouse Pointer"),
    "rightClick":  ("icon_images/rightClick.png",  "Right Mouse Click"),
}

gesture_icons = {}
icon_size = (50, 50)  # icon display size
for gesture, (filename, desc) in gesture_icons_info.items():
    if os.path.exists(filename):
        icon = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if icon is not None:
            icon = cv2.resize(icon, icon_size, interpolation=cv2.INTER_AREA)
            gesture_icons[gesture] = icon
        else:
            print(f"Warning: Could not read file {filename}")
            gesture_icons[gesture] = None
    else:
        print(f"Warning: File {filename} does not exist.")
        gesture_icons[gesture] = None

# -------------------------
# 2) Webcam Initialization
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

roi_top, roi_right, roi_bottom, roi_left = 20, 300, 350, 640

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)
cap.set(cv2.CAP_PROP_CONTRAST, 50)

# Thread-safe queue for frames
frame_queue = queue.Queue(maxsize=2)
def frame_capture(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

capture_thread = threading.Thread(target=frame_capture, args=(cap, frame_queue))
capture_thread.daemon = True
capture_thread.start()

# -------------------------
# 3) Background Subtraction Variables
# -------------------------
bg = None
bg_frames = 30
bg_counter = 0
gesture_buffer = collections.deque(maxlen=7)

# -------------------------
# 5) Kalman Filter Setup
# -------------------------
# Initialize Kalman filter with state size = 4 (x, y, vx, vy) and measurement size = 2 (x, y)
kalman = cv2.KalmanFilter(4, 2)

# Measurement matrix (only observes position, not velocity)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

# Transition matrix (includes velocity terms)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],  # x' = x + vx
                                    [0, 1, 0, 1],  # y' = y + vy
                                    [0, 0, 1, 0],  # vx' = vx
                                    [0, 0, 0, 1]], np.float32)  # vy' = vy

# Process noise covariance (controls smoothness; increase slightly for faster response)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05

# Measurement noise covariance (reduces noise in position updates)
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.03

# Error covariance (adjusts initial uncertainty)
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

# Initialize state with an arbitrary value (avoid instability at start)
kalman.statePre = np.array([[0], [0], [0], [0]], dtype=np.float32)

# State Variables
mode = None
locked = False
in_drag = False

# Continuous Movement Variables
prev_cx, prev_cy = None, None
velocity_x, velocity_y = 0, 0
vel_alpha = 0.2
sudden_change_threshold = 40
small_threshold = 5
speed_factor = 5

# -------------------------
# 5) Helper Functions
# -------------------------
def run_avg(image, accumWeight=0.5):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25, min_area=1000):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological ops
    kernel = np.ones((3,3), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Canny edges
    edges = cv2.Canny(thresholded, 50, 150)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    segmented = max(contours, key=cv2.contourArea)
    if cv2.contourArea(segmented) < min_area:
        return None, None
    return thresholded, (segmented, edges)

def move_cursor(dx, dy):
    pyautogui.moveRel(int(dx * speed_factor), int(dy * speed_factor), duration=0)

def perform_mouse_action(gesture):
    if gesture == "doubleClick":
        pyautogui.doubleClick()
    elif gesture == "leftClick":
        pyautogui.click()
    elif gesture == "rightClick":
        pyautogui.rightClick()

def kalman_predict_and_correct(cx, cy):
    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
    kalman.correct(measurement)
    prediction = kalman.predict()
    cx_smooth, cy_smooth = prediction[0][0], prediction[1][0]
    return cx_smooth, cy_smooth

# -------------------------
# 6) Main Loop
# -------------------------
while True:
    if not frame_queue.empty():
        frame = frame_queue.get()
    else:
        continue

    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    height, width = clone.shape[:2]

    # ROI
    roi = clone[roi_top:roi_bottom, roi_right:roi_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Background Calibration
    if bg_counter < bg_frames:
        run_avg(gray)
        bg_counter += 1
        cv2.putText(clone, f"Calibrating background: {bg_counter}/{bg_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(clone, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
        cv2.imshow("Gesture Recognition", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    # Hand Segmentation & Gesture Prediction
    thresholded, seg_result = segment(gray)
    current_gesture = None
    cx = cy = None

    if seg_result is not None:
        segmented, edges = seg_result

        # Draw edges in green on the ROI portion of the feed
        roi[edges > 0] = [0, 255, 0]

        # Compute centroid
        M = cv2.moments(segmented)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(roi, (cx, cy), 5, (255, 0, 0), -1)
        else:
            cx, cy = None, None

        # Prepare thresholded image for model
        thresh_copy = cv2.resize(thresholded, (200, 200))
        thresh_copy = thresh_copy.astype("float32") / 255.0
        thresh_copy = np.expand_dims(thresh_copy, axis=-1)
        thresh_copy = np.expand_dims(thresh_copy, axis=0)

        predictions = model.predict(thresh_copy)
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        if confidence > 0.75:
            gesture_buffer.append(predicted_class)
        else:
            gesture_buffer.append(-1)

        most_common, count = collections.Counter(gesture_buffer).most_common(1)[0]
        if most_common != -1 and count >= 3:
            current_gesture = class_labels[most_common]

        cv2.imshow("Thresholded", thresholded)
    else:
        cv2.putText(clone, "No hand detected", (width - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mode & Gesture Logic
    if current_gesture == "moveCursor":
        mode = "moveCursor"
        locked = False
    elif current_gesture == "drag":
        if not in_drag:
            mode = "drag"
            in_drag = True
            pyautogui.mouseDown()
        else:
            mode = "drag"
    elif current_gesture == "drop":
        if in_drag:
            pyautogui.mouseUp()
            in_drag = False
            mode = "drop"
            locked = False
        elif locked:
            locked = False
            mode = "drop"
        else:
            mode = None
    elif current_gesture in ["leftClick", "rightClick", "doubleClick"]:
        if not locked:
            mode = current_gesture
            perform_mouse_action(current_gesture)
            locked = True
    else:
        if not locked and not in_drag:
            mode = None


    # Continuous Movement with Kalman Filter
    if mode in ["moveCursor", "drag"] and cx is not None and cy is not None:
        # Get smoothed coordinates from Kalman
        cx_smooth, cy_smooth = kalman_predict_and_correct(cx, cy)

        if prev_cx is not None and prev_cy is not None:
            dx = cx_smooth - prev_cx
            dy = cy_smooth - prev_cy
            if abs(dx) > sudden_change_threshold or abs(dy) > sudden_change_threshold:
                dx, dy = 0, 0
            if abs(dx) >= small_threshold or abs(dy) >= small_threshold:
                velocity_x = vel_alpha * dx + (1 - vel_alpha) * velocity_x
                velocity_y = vel_alpha * dy + (1 - vel_alpha) * velocity_y
        prev_cx, prev_cy = cx_smooth, cy_smooth
        move_cursor(velocity_x, velocity_y)
    else:
        velocity_x, velocity_y = 0, 0


    # ----------------------------------------
    # Display Icons + Move Mode Text Right
    # ----------------------------------------
    x_icon = 10
    y_icon = 10
    spacing = 70

    for i, gesture in enumerate(class_labels):
        icon = gesture_icons.get(gesture, None)
        _, gesture_desc = gesture_icons_info[gesture]
        y_offset = y_icon + i * spacing

        if icon is not None:
            h, w_ = icon.shape[:2]
            roi_icon = clone[y_offset:y_offset+h, x_icon:x_icon+w_]
            if icon.shape[2] == 4:  # alpha channel
                icon_bgr = icon[:,:,:3]
                alpha_mask = icon[:,:,3] / 255.0
                for c in range(3):
                    roi_icon[:,:,c] = (alpha_mask * icon_bgr[:,:,c] +
                                       (1 - alpha_mask) * roi_icon[:,:,c])
            else:
                clone[y_offset:y_offset+h, x_icon:x_icon+w_] = icon

            cv2.putText(clone, gesture_desc, (x_icon + w_ + 10, y_offset + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(clone, f"{gesture} - {gesture_desc}",
                        (x_icon, y_offset + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(clone, f"Mode: {mode}", (width - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.rectangle(clone, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", clone)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        bg = None
        bg_counter = 0
        mode = None
        locked = False
        in_drag = False
        velocity_x, velocity_y = 0, 0

cap.release()
cv2.destroyAllWindows()
