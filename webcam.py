from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import cv2
import numpy as np
from PIL import Image
import os

# Load the model and image processor from Hugging Face
model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Path to your input image (replace with your image path)
input_image_path = "/Users/manvith/Downloads/projects/interdisp/leaf.JPG"  # e.g., "leaf.jpg"

# Check if the input image exists
if not os.path.exists(input_image_path):
    print(f"Error: Image file {input_image_path} not found.")
    exit()

# Load and preprocess the image
image = cv2.imread(input_image_path)
if image is None:
    print(f"Error: Could not load image {input_image_path}.")
    exit()

# Convert BGR (OpenCV) to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to PIL Image for processing
pil_image = Image.fromarray(rgb_image)

# Preprocess the image for the model
inputs = processor(images=pil_image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    confidence = torch.softmax(logits, dim=1)[0, predicted_class_idx].item()

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2%}")

# Overlay text with smaller font and wrap if needed
font_scale = 0.5  # Smaller font size
font_thickness = 1
text = f"Prediction: {predicted_class} ({confidence:.2%})"

# Get text size to check if it fits
(text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

# If text is too wide, split into two lines
max_width = image.shape[1] - 20  # Image width minus padding
if text_width > max_width:
    # Split text into two parts (prediction and confidence)
    line1 = f"Prediction: {predicted_class}"
    line2 = f"Confidence: {confidence:.2%}"
    cv2.putText(image, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(image, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
else:
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

# Save the output image
output_image_path = f"output_{os.path.basename(input_image_path)}"
cv2.imwrite(output_image_path, image)
print(f"Output image saved as {output_image_path}")