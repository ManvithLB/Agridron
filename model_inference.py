import torch
import cv2
import numpy as np
from PIL import Image
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load model and processor
model_name = "Diginsa/Plant-Disease-Detection-Project"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def run_inference(input_image_path, confidence_threshold=0.95):
    # Check if image exists
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Image file {input_image_path} not found.")
    
    # Load and preprocess image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not load image {input_image_path}.")
    
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Preprocess for model
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = model.config.id2label[predicted_class_idx]
            confidence = probabilities[predicted_class_idx].item()
            
            # Apply 95% confidence threshold
            if predicted_class.lower() != "healthy" and confidence < confidence_threshold:
                predicted_class = "Healthy"
                confidence = 1.0 - confidence  # Adjust confidence to reflect healthy prediction
        
        # Overlay text on image
        font_scale = 0.5
        font_thickness = 1
        text = f"Prediction: {predicted_class} ({confidence:.2%})"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        max_width = image.shape[1] - 20
        if text_width > max_width:
            line1 = f"Prediction: {predicted_class}"
            line2 = f"Confidence: {confidence:.2%}"
            cv2.putText(image, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            cv2.putText(image, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        else:
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        
        # Save output image
        output_image_path = f"output_{os.path.basename(input_image_path)}"
        cv2.imwrite(output_image_path, image)
        
        return predicted_class, confidence, output_image_path
    finally:
        # Release OpenCV resources
        cv2.destroyAllWindows()