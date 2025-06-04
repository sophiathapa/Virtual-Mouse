import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import collections
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------------
# Load the trained model and class labels
# -------------------------------
model = tf.keras.models.load_model('gesture_recognition_model.h5')
print("Model loaded successfully.")

# Class labels (order must match your trained model)
class_labels = ["doubleClick", "drag", "drop", "leftClick", "moveCursor", "rightClick"]

# -------------------------------
# Initialize Webcam & Set Camera Properties
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)
cap.set(cv2.CAP_PROP_CONTRAST, 32)

# -------------------------------
# Define a Larger ROI for Hand Detection
# -------------------------------
roi_top, roi_right, roi_bottom, roi_left = 20, 300, 350, 640

# -------------------------------
# Background subtraction & Gesture Smoothing
# -------------------------------
bg = None
bg_captured = False
gesture_buffer = collections.deque(maxlen=5)  # For majority vote smoothing

# -------------------------------
# State Variables for Gesture-Controlled Actions
# -------------------------------
last_mode = None          # Last recognized continuous gesture mode
one_time_triggered = False  # Flag to prevent repeated one-time actions
in_drag_mode = False      # Flag indicating if a drag is currently active

# For continuous cursor movement
prev_cx, prev_cy = None, None   # Previous centroid coordinates of the hand
last_direction = (0, 0)         # Last movement vector (dx, dy)
sudden_change_threshold = 40    # Ignore abnormally large jumps
small_threshold = 5             # Minimal movement required to update direction
speed_factor = 2                # Amplify the relative movement

# -------------------------------
# Helper Functions
# -------------------------------
def run_avg(image, accumWeight=0.5):
    """Compute the running average for background subtraction."""
    global bg
    if bg is None:
        bg = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25):
    """Segment the hand region from the background using absolute difference."""
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        segmented = max(contours, key=cv2.contourArea)
        return thresholded, segmented

def move_cursor(dx, dy):
    """Move the mouse cursor relative to its current position.
       dx, dy are multiplied by speed_factor."""
    pyautogui.moveRel(dx * speed_factor, dy * speed_factor, duration=0.1)

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect and clone for display
    frame = cv2.flip(frame, 1)
    clone = frame.copy()

    # Extract the ROI for gesture detection
    roi = clone[roi_top:roi_bottom, roi_right:roi_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # -------------------------------
    # Background Calibration
    # -------------------------------
    if not bg_captured:
        run_avg(gray)
        cv2.putText(clone, "Calibrating background...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        bg_captured = True
        cv2.rectangle(clone, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
        cv2.imshow("Gesture Recognition", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    # -------------------------------
    # Hand Segmentation and Gesture Prediction
    # -------------------------------
    hand = segment(gray)
    current_gesture = None
    cx = cy = None  # Hand centroid
    if hand is not None:
        thresholded, segmented = hand

        # Compute centroid of the hand for movement calculations
        M = cv2.moments(segmented)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(roi, (cx, cy), 5, (255, 0, 0), -1)
        else:
            cx, cy = None, None

        # Prepare the thresholded image for gesture prediction
        thresh_copy = cv2.resize(thresholded, (128, 128))
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

        # Majority vote among the buffered predictions
        most_common, count = collections.Counter(gesture_buffer).most_common(1)[0]
        if most_common != -1 and count >= 3:
            current_gesture = class_labels[most_common]
        else:
            current_gesture = None

        cv2.imshow("Thresholded", thresholded)
    else:
        current_gesture = None

    # -------------------------------
    # Detect Gesture Change (State Transition)
    # -------------------------------
    if current_gesture != last_mode:
        # If leaving drag mode, drop the item (simulate mouseUp)
        if last_mode == "drag" and in_drag_mode:
            pyautogui.mouseUp()
            in_drag_mode = False
        # Reset one-time trigger for one-shot actions
        one_time_triggered = False
        last_mode = current_gesture

    # -------------------------------
    # Execute Action Based on Current Gesture
    # -------------------------------
    if current_gesture == "moveCursor":
        # Update the movement direction based on hand centroid differences
        if cx is not None and cy is not None and prev_cx is not None and prev_cy is not None:
            dx = cx - prev_cx
            dy = cy - prev_cy
            # Only update if the change is within a reasonable limit and is significant
            if abs(dx) <= sudden_change_threshold and abs(dy) <= sudden_change_threshold:
                if abs(dx) > small_threshold or abs(dy) > small_threshold:
                    last_direction = (dx, dy)
        # Continuously move the cursor in the last detected direction
        move_cursor(last_direction[0], last_direction[1])
        cv2.putText(clone, "Mode: moveCursor", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    elif current_gesture == "drag":
        # Start dragging if not already in drag mode
        if not in_drag_mode:
            pyautogui.mouseDown()
            in_drag_mode = True
        # Update the movement direction (similar to moveCursor)
        if cx is not None and cy is not None and prev_cx is not None and prev_cy is not None:
            dx = cx - prev_cx
            dy = cy - prev_cy
            if abs(dx) <= sudden_change_threshold and abs(dy) <= sudden_change_threshold:
                if abs(dx) > small_threshold or abs(dy) > small_threshold:
                    last_direction = (dx, dy)
        move_cursor(last_direction[0], last_direction[1])
        cv2.putText(clone, "Mode: drag", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    elif current_gesture in ["leftClick", "rightClick", "doubleClick"]:
        # Trigger the one-time click action if not already triggered
        if not one_time_triggered:
            if current_gesture == "leftClick":
                pyautogui.click()
            elif current_gesture == "rightClick":
                pyautogui.rightClick()
            elif current_gesture == "doubleClick":
                pyautogui.doubleClick()
            one_time_triggered = True
        cv2.putText(clone, f"Mode: {current_gesture}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        # When no recognized gesture or a non-continuous gesture is active, do nothing.
        cv2.putText(clone, "Mode: None", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Update previous centroid for next frame (if available)
    if cx is not None and cy is not None:
        prev_cx, prev_cy = cx, cy

    # Draw ROI rectangle and display the main frame
    cv2.rectangle(clone, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", clone)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Recalibrate background and reset state
        bg = None
        bg_captured = False
        last_mode = None
        one_time_triggered = False
        in_drag_mode = False

# Release the webcam and close windows.
cap.release()
cv2.destroyAllWindows()
