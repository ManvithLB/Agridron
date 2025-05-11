import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import pyrebase
import time
from model_inference import run_inference
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import logging
import random
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load environment variables
load_dotenv(dotenv_path=".env-flask")
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in .env-flask")
    raise ValueError("OPENAI_API_KEY is required")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

# Firebase configuration
firebase_config = {
    "apiKey": google_api_key,
    "authDomain": "agriinter-6292d.firebaseapp.com",
    "projectId": "agriinter-6292d",
    "storageBucket": "agriinter-6292d.firebasestorage.app",
    "messagingSenderId": "725950782697",
    "appId": "1:725950782697:web:38f17d88dbe5cbf485a478",
    "measurementId": "G-S0T7NVN5MG",
    "databaseURL": "https://agriinter-6292d-default-rtdb.firebaseio.com",
    "serviceAccount": "firebase_config.json"
}

try:
    logger.info("Initializing Firebase app")
    firebase = pyrebase.initialize_app(firebase_config)
    storage = firebase.storage()
    db = firebase.database()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {str(e)}")
    raise

# Global variables
capturing = False
capture_data = []
TESTING_MODE = True

# Drone square path state
drone_state = {
    'step': 0,  # Current step (0 to 39 for 10 steps per side)
    'base_lat': None,  # Initial latitude
    'base_lon': None,  # Initial longitude
    'side_length': 0.02,  # Square side length in degrees (~2.2 km)
    'steps_per_side': 10,  # Number of steps per side
    'step_size': 0.002  # Step size in degrees (side_length / steps_per_side)
}

def reset_drone_state(base_lat, base_lon):
    """Reset drone state with new base coordinates."""
    drone_state['step'] = 0
    drone_state['base_lat'] = base_lat
    drone_state['base_lon'] = base_lon
    logger.info(f"Drone state reset: base_lat={base_lat}, base_lon={base_lon}")

def get_square_path_coords():
    """Calculate drone coordinates for the current step in a square path."""
    step = drone_state['step']
    base_lat = drone_state['base_lat']
    base_lon = drone_state['base_lon']
    step_size = drone_state['step_size']
    steps_per_side = drone_state['steps_per_side']

    # Determine side (0: North, 1: East, 2: South, 3: West)
    side = step // steps_per_side
    side_step = step % steps_per_side

    if side == 0:  # North: increase latitude
        lat = base_lat + side_step * step_size
        lon = base_lon
    elif side == 1:  # East: increase longitude
        lat = base_lat + steps_per_side * step_size
        lon = base_lon + side_step * step_size
    elif side == 2:  # South: decrease latitude
        lat = base_lat + (steps_per_side - side_step) * step_size
        lon = base_lon + steps_per_side * step_size
    else:  # West: decrease longitude
        lat = base_lat
        lon = base_lon + (steps_per_side - side_step) * step_size

    # Increment step, reset to 0 after completing the square (40 steps)
    drone_state['step'] = (step + 1) % (4 * steps_per_side)
    logger.info(f"Drone step {step}: side={side}, lat={lat:.6f}, lon={lon:.6f}")

    return lat, lon

def generate_insights(results):
    if not results:
        logger.error("No results provided for insights generation")
        return "<strong>No insights available</strong>"

    # Count diseases
    disease_counts = {}
    print(results)
    healthy_count = 0
    for point in results:
        prediction = point['prediction'].lower()
        if prediction == 'healthy':
            healthy_count += 1
        else:
            disease_counts[prediction] = (disease_counts.get(prediction, 0) + 1)

    try:
        # Call OpenAI /v1/chat/completions
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an agricultural expert helping farmers."},
                {"role": "user", "content": f"Generate insights for a farmer based on plant disease data: {healthy_count} healthy plants, {disease_counts}. Provide treatment recommendations.If healthy, suggest regular maintenance."}
            ],
            max_tokens=200
        )
        insights = response.choices[0].message.content.strip()
        logger.info("Generated insights from OpenAI: %s", insights)
        return insights
    except Exception as e:
        logger.error("Failed to generate insights from OpenAI: %s", str(e))
        # Fallback static insights
        summary = f"Analyzed {len(results)} plants: {healthy_count} healthy"
        if disease_counts:
            summary += ", " + ", ".join([f"{count} with {disease.capitalize()}" for disease, count in disease_counts.items()])
        else:
            summary += ", no diseases detected."
        
        advice = "Continue regular maintenance for healthy plants with balanced fertilizer (NPK 10-10-10)."
        for disease, count in disease_counts.items():
            coords = next(p for p in results if p['prediction'].lower() == disease)
            fertilizer = {
                'apple black rot': 'Apply copper-based fungicide (e.g., Bordeaux mixture) weekly until symptoms clear. Remove infected fruit/leaves.',
                'default': 'Consult a local agronomist for specific fungicides or treatments.'
            }.get(disease, 'default')
            advice += f" For {count} plant(s) with {disease.capitalize()} at [{coords['latitude']:.4f}, {coords['longitude']:.4f}], {fertilizer}"

        advice_html = advice.replace('\n', '<br>')
        insights = f"<strong>Farmer Insights:</strong><br>{summary}<br><br><strong>Recommendations:</strong><br>{advice_html}"
        return insights

@app.route('/')
def index():
    logger.info("Serving index.html")
    return render_template('index.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capturing
    capturing = True
    capture_data.clear()
    logger.info("Started capturing images")
    return jsonify({"status": "started"})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global capturing
    capturing = False
    logger.info("Stopped capturing images")
    
    results = []
    for data in capture_data:
        img_data, lat, lon, timestamp = data
        logger.info(f"Processing image for timestamp {timestamp}")
        
        try:
            img_bytes = base64.b64decode(img_data.split(',')[1])
            img = Image.open(BytesIO(img_bytes))
            img.save("temp.jpg")
            logger.info(f"Saved temporary image for timestamp {timestamp}")
            
            prediction, confidence, output_path = run_inference("temp.jpg")
            logger.info(f"Inference completed: {prediction} ({confidence:.2%})")
            
            output_filename = f"output_{timestamp}.jpg"
            storage.child(f"images/{output_filename}").put(output_path)
            image_url = storage.child(f"images/{output_filename}").get_url(None)
            logger.info(f"Uploaded output image to Firebase Storage: {output_filename}")
            
            results.append({
                "prediction": prediction,
                "confidence": confidence,
                "image_url": image_url,
                "latitude": lat,
                "longitude": lon,
                "timestamp": timestamp
            })
        except Exception as e:
            logger.error(f"Error processing image for timestamp {timestamp}: {str(e)}")
            continue
    
    for result in results:
        try:
            db.child("results").push(result)
            logger.info(f"Saved result to Realtime Database: {result['timestamp']}")
        except Exception as e:
            logger.error(f"Error saving result to Realtime Database: {str(e)}")
    
    insights = generate_insights(results)
    
    logger.info(f"Returning {len(results)} results with insights")
    return jsonify({"status": "stopped", "results": results, "insights": insights})

@app.route('/capture', methods=['POST'])
def capture():
    if not capturing:
        logger.warning("Capture attempted but not capturing")
        return jsonify({"status": "not_capturing"})
    
    data = request.json
    if not data or 'image' not in data or 'latitude' not in data or 'longitude' not in data:
        logger.error("Invalid capture data received")
        return jsonify({"status": "error", "message": "Invalid data"}), 400
    
    image_data = data['image']
    latitude = data['latitude']
    longitude = data['longitude']
    
    if TESTING_MODE:
        # Initialize drone state on first capture
        if drone_state['base_lat'] is None or drone_state['base_lon'] is None:
            reset_drone_state(latitude, longitude)
        
        # Get square path coordinates
        latitude, longitude = get_square_path_coords()
        logger.info(f"Applied square path: lat={latitude:.6f}, lon={longitude:.6f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Capturing image with timestamp {timestamp}, lat: {latitude}, lon: {longitude}")
    
    try:
        capture_data.append((image_data, latitude, longitude, timestamp))
        logger.info(f"Stored capture data in memory for timestamp {timestamp}")
        
        filename = f"image_{timestamp}.jpg"
        img_bytescdn:0
        img_bytes = base64.b64decode(image_data.split(',')[1])
        storage.child(f"images/{filename}").put(img_bytes)
        logger.info(f"Uploaded image to Firebase Storage: {filename}")
        
        db.child("captures").push({
            "filename": filename,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": timestamp
        })
        logger.info(f"Saved metadata to Realtime Database: {filename}")
        
        return jsonify({"status": "captured"})
    except Exception as e:
        logger.error(f"Error during capture for timestamp {timestamp}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)