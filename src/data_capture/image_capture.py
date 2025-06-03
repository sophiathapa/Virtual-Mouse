import cv2
import imutils
import numpy as np

bg = None

# Function - To find the running average over the background
def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

# Function - To segment the region of hand in the image
def segment(image, threshold=30):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None
    else:
        segmented = max(cnts, key=lambda x: cv2.contourArea(x))
        return (thresholded, segmented)




# Main function
if __name__ == "__main__":
    accumWeight = 0.2
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 210, 550
    num_frames = 0
    imageNumber = 0
    thresholded = None  # Initialize thresholded to avoid NameError

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        height, width = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print(">>> Please wait! Calibrating...")
            elif num_frames == 29:
                print(">>> Calibration successful...")
        else:
            hand = segment(gray)

            
            if hand is not None:
                thresholded, segmented = hand
                cv2.imshow("Thresholded", thresholded)

        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('s') and thresholded is not None:
            # Save image when 's' is pressed
            directory = f"C:\\Users\\Dell\\OneDrive\\Pictures\\Desktop\\Major-Project\\data-collection\\images\\data({imageNumber}).jpg" 
            cv2.imwrite(directory, thresholded)
            imageNumber += 1
            print(f"Image saved at {directory}")

        if keypress == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


