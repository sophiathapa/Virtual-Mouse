import os
import shutil
from sklearn.model_selection import train_test_split

# Path to the dataset directory
dataset_dir = "C:/Users/Dell/OneDrive/Pictures/Desktop/Major-Project/dataset_2k"

# Create directories for train, validation, and test sets
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# List of gesture directories
gestures = ["doubleClick", "drag", "drop", "leftClick", "moveCursor", "rightClick"]

for gesture in gestures:
    gesture_path = os.path.join(dataset_dir, gesture)
    images = [os.path.join(gesture_path, img) for img in os.listdir(gesture_path)]
    
    # Split images into train, validation, and test sets (75% train, 15% val, 15% test)
    train_images, temp_images = train_test_split(images, test_size=0.30, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)
    
    # Create gesture subdirectories in train, val, and test folders
    train_gesture_dir = os.path.join(train_dir, gesture)
    val_gesture_dir = os.path.join(val_dir, gesture)
    test_gesture_dir = os.path.join(test_dir, gesture)

    for directory in [train_gesture_dir, val_gesture_dir, test_gesture_dir]:
        os.makedirs(directory, exist_ok=True)

    # Copy images to the respective directories
    for img in train_images:
        shutil.copy(img, os.path.join(train_gesture_dir, os.path.basename(img)))
    for img in val_images:
        shutil.copy(img, os.path.join(val_gesture_dir, os.path.basename(img)))
    for img in test_images:
        shutil.copy(img, os.path.join(test_gesture_dir, os.path.basename(img)))

print("Dataset has been successfully split into train (75%), validation (15%), and test (15%) sets with gesture subfolders.")