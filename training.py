import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = 'C:/Users/Balaji/Documents/Project/New folder'  # Path to your dataset containing the 4 folders
output_dir = 'C:/Users/Balaji/Documents/Project'  # Directory to store the split dataset

# Define the ratio
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create directories for the split dataset
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'validation')
test_dir = os.path.join(output_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get the 4 class folders (assuming your dataset is organized in folders per class)
class_labels = os.listdir(dataset_dir)

# Loop through each class folder and split images
for label in class_labels:
    class_path = os.path.join(dataset_dir, label)
    images = os.listdir(class_path)
    
    # Split into training and temp (val+test)
    train_images, temp_images = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
    
    # Further split temp into validation and test
    val_images, test_images = train_test_split(temp_images, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)
    
    # Create class folders in train, val, and test directories
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    # Copy images to respective directories
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, label, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, label, img))

    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, label, img))

print("Dataset split complete!")