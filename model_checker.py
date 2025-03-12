import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

# Load your trained model
model_path = r(Model PAth)
model = load_model(model_path)

# Initialize Tkinter root
root = Tk()
root.withdraw()  # Hide the main window

# Ask user to select multiple image files
file_paths = filedialog.askopenfilenames(
    title="Select Skin Cancer Images",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

# Check if any files were selected
if not file_paths:
    print("No files selected. Exiting...")
    exit()

# Process selected images
for file_path in file_paths:
    try:
        # Load and preprocess image
        img = image.load_img(file_path, target_size=(224, 224))  # Adjust target_size to match model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize if your model expects normalized inputs

        # Make prediction
        prediction = model.predict(img_array)
        
        # Get confidence score and class
        confidence = prediction[0][0]  # Adjust index based on your model's output shape
        predicted_class = "Malignant" if confidence > 0.5 else "Benign"  # Modify based on your classes

        # Display results
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(122)
        plt.text(0.1, 0.6, f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}", fontsize=12)
        plt.axis('off')
        
        plt.suptitle(os.path.basename(file_path))
        plt.show()

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

print("Prediction completed for all selected images.")