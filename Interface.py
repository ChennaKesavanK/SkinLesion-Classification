import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model = load_model('C:/Users/chenna kesavan/OneDrive/Desktop/skin cancer detection/best_model.keras')

# Class names (replace with your classes)
class_names = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma','Actinic Keratosis', 'Dermatofibroma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Vascular lesion']

# Function to preprocess the image and predict
def predict_image():
    # Open a file dialog to select the image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "")])
    if not file_path:
        return  # Exit if no file is selected

    # Load and preprocess the image
    img = load_img(file_path, target_size=(224, 224))  # Adjust target_size as per your model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image (if required)

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # Display the image and prediction
    img_display = Image.open(file_path)
    img_display.thumbnail((300, 300))  # Resize image for display
    img_display = ImageTk.PhotoImage(img_display)
    img_label.config(image=img_display)
    img_label.image = img_display

    result_label.config(text=f"Predicted Class: {predicted_class}")

# Initialize Tkinter window
window = tk.Tk()
window.title("Image Classifier")
window.geometry("500x600")

# Create widgets
header_label = tk.Label(window, text="Image Classifier", font=("Arial", 20))
header_label.pack(pady=10)

img_label = tk.Label(window)
img_label.pack(pady=20)

result_label = tk.Label(window, text="Prediction will appear here", font=("Arial", 14))
result_label.pack(pady=10)

predict_button = tk.Button(window, text="Select and Predict Image", command=predict_image, font=("Arial", 14))
predict_button.pack(pady=20)

# Run the Tkinter event loop
window.mainloop()


