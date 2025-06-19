# # pip install fastapi uvicorn onnxruntime opencv-python python-multipart
# # pip install websockets

import numpy as np
import cv2
import onnxruntime as ort
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile, Form
from fastapi.responses import JSONResponse
import io
from PIL import Image
import base64
import logging
from firebase_admin import credentials, db
import firebase_admin
import pandas as pd
from typing import Dict, Any, List, Union, Tuple
from sklearn.ensemble import IsolationForest
from fastapi.responses import StreamingResponse
import time
import requests
from datetime import datetime, timezone

# Initialize FastAPI app
app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("credentials.json") #TODO: use secret manager or environment variable for credentials
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sih2024-559e6-default-rtdb.firebaseio.com/'
})

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ONNX model
onnx_model_path = "bestm.onnx"
try:
    session = ort.InferenceSession(onnx_model_path)
    logger.info("ONNX model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ONNX model: {e}")

# Define the threshold for detection and NMS IoU
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Assuming 29 classes for the model
CLASSES = ['Aluminium foil', 'Bottle cap', 'Broken glass', 'Cigarette', 'Clear plastic bottle', 'Crisp packet', 
           'Cup', 'Drink can', 'Food Carton', 'Food container', 'Food waste', 'Garbage bag', 'Glass bottle', 
           'Lid', 'Other Carton', 'Other can', 'Other container', 'Other plastic bottle', 'Other plastic wrapper', 
           'Other plastic', 'Paper bag', 'Paper', 'Plastic bag wrapper', 'Plastic film', 'Pop tab', 
           'Single-use carrier bag', 'Straw', 'Styrofoam piece', 'Unlabeled litter']

# Threshold-based alert settings
DETECTION_THRESHOLD = 20  # Spillage area threshold in percent
ALERT_COUNT_THRESHOLD = 10  # Number of times threshold is exceeded to trigger an alert
detection_counter = 0  # Counts consecutive threshold exceedances
alert_triggered = False  # Track if alert has been triggered

# Function to perform non-maximum suppression (NMS)
def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, iou_threshold)
    if len(indices) > 0:
        return indices.flatten()
    else:
        return []

# Function to calculate area of detected trash
def calculate_spillage_area(bboxes, frame_size):
    frame_area = frame_size[0] * frame_size[1]
    trash_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes)
    spillage_percentage = (trash_area / frame_area) * 100
    return spillage_percentage

# Function to process frame through ONNX model and get bounding boxes
def detect_trash(frame):
    try:
        img = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Transpose the image from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)+
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Run the model inference
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: img})

        # Extract the model output
        output = result[0]  # [1, 25200, 34]
        output = np.squeeze(output, axis=0)  # Remove batch dimension, shape becomes [25200, 34]

        bboxes = []
        confidences = []
        class_ids = []
        detected_classes = []

        for detection in output:
            # Extract the box coordinates and confidence
            x_center, y_center, width, height = detection[:4]
            confidence = detection[4]  # Objectness score
            class_scores = detection[5:]  # Class probabilities

            if confidence > CONFIDENCE_THRESHOLD:
                # Get the class with the highest score
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]

                if class_confidence > CONFIDENCE_THRESHOLD:
                    # Convert the center x, y, width, height to x1, y1, x2, y2 (top-left and bottom-right)
                    x1 = int((x_center - width / 2))
                    y1 = int((y_center - height / 2))
                    x2 = int((x_center + width / 2))
                    y2 = int((y_center + height / 2))

                    bboxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    detected_classes.append(CLASSES[class_id])

        # Perform non-maximum suppression to filter overlapping boxes
        indices = non_max_suppression(bboxes, confidences, NMS_THRESHOLD)

        # Only keep the boxes after NMS
        final_bboxes = [bboxes[i] for i in indices]
        final_confidences = [confidences[i] for i in indices]
        final_class_ids = [class_ids[i] for i in indices]
        final_detected_classes = [detected_classes[i] for i in indices]

        return final_bboxes, final_confidences, final_class_ids, final_detected_classes
    except Exception as e:
        logger.error(f"Error in detect_trash function: {e}")
        return [], [], [], []

# Function to draw bounding boxes on the frame
def draw_boxes(frame, bboxes, class_ids, confidences):
    try:
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            class_id = class_ids[i]
            confidence = confidences[i]

            # Ensure coordinates are within the frame bounds
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(0, min(x2, frame.shape[1] - 1))
            y2 = max(0, min(y2, frame.shape[0] - 1))

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label and confidence
            label = f"{CLASSES[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame
    except Exception as e:
        logger.error(f"Error in draw_boxes function: {e}")
        return frame
    
def update_db(postoffice_id, place, time, class_id):
    try:
        if not class_id:
            # Handle compliant case
            ref = db.reference(f'/postOffices/{postoffice_id}/compliant')
            data = ref.get() or 0
            ref.set(data + 1)
            return

        # Handle non-compliant case
        ref = db.reference(f'/postOffices/{postoffice_id}/non-compliant')
        data = ref.get() or 0
        ref.set(data + 1)

        # Push detection to timetable data
        ref = db.reference(f'/postOffices/{postoffice_id}/detectionTimeTableData')
        for i in class_id:
            ref.push({
                'place': place,
                'time': time,
                'type': CLASSES[i],
            })

        # Update detection count by date
        date = time.split("T")[0]
        ref = db.reference(f'/postOffices/{postoffice_id}/garbageDetectionData/{date}')
        current_data = ref.get() or {}
        detections = current_data.get('detections', 0)
        ref.update({'detections': detections + 1})

        # Update garbage type data
        ref = db.reference(f'/postOffices/{postoffice_id}/garbageTypeData')
        current_data = ref.get() or {}
        for i in class_id:
            current_data[i]['frequency'] += 1
        ref.set(current_data)

        logger.info(f"Database updated successfully for postoffice_id: {postoffice_id}")

    except Exception as e:
        logger.error(f"Error updating database for postoffice_id {postoffice_id}: {e}")
    

def funny(frame, postoffice_id, place, time):
    try:
        if frame is None:
            return False

        # Perform trash detection
        input_size = (640, 640)
        frame = cv2.resize(frame, input_size)
        bboxes, confidences, class_ids, detected_classes = detect_trash(frame)
        if(postoffice_id != ""):
            update_db(postoffice_id, place, time, class_ids)

        if detected_classes == []:
            return False

        # Print detected classes to terminal
        logger.info(f"Detected Classes: {detected_classes}")

        # Calculate spillage percentage
        spillage_percentage = calculate_spillage_area(bboxes, frame.shape[:2])
        logger.info(f"Spillage Percentage: {spillage_percentage}")

        # Draw boxes on the frame
        frame = draw_boxes(frame, bboxes, class_ids, confidences)

        return frame
    except Exception as e:
        logger.error(f"Error in funny function: {e}")
        return False

@app.get("/")
async def get():
    return HTMLResponse(content="<h1>Welcome to the Trash Detection API</h1>")

@app.post("/new-file/")
async def new_file(postoffice_id: str = Form(""), place: str = Form("")):
    logger.info(f"Postoffice ID: {postoffice_id} Place: {place}")
    template_code = """
import cv2
import time
import requests
from datetime import datetime, timezone

API_URL = "https://modelapi-pniz.onrender.com/detect-trash/"  
CAMERA_INDEX = 1

postoffice_id = "{postoffice_id}"
place = "{place}"

def capture_image():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while not cap.isOpened():
        print("Waiting for camera to warm up...")
        time.sleep(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from camera")
    _, img_encoded = cv2.imencode('.jpg', frame)
    return img_encoded.tobytes()

def send_image(image_bytes):
    files = {{'file': ('image.jpg', image_bytes, 'image/jpeg')}}
    time_str = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    data = {{
        'postoffice_id': postoffice_id,
        'place': place,
        'time': time_str
    }}
    response = requests.post(API_URL, files=files, data=data)
    return response

if __name__ == "__main__":
    while True:
        try:
            img_bytes = capture_image()
            resp = send_image(img_bytes)
            print(f"Sent image, status code: {{resp.status_code}}")
        except Exception as e:
            print(f"Error: {{e}}")
        time.sleep(300)  # 5 minutes
"""

    code_content = template_code.format(postoffice_id=postoffice_id, place=place)
    return StreamingResponse(io.BytesIO(code_content.encode()), media_type="text/x-python", headers={"Content-Disposition": "attachment; filename=client.py"})

@app.post("/detect-trash/")
async def detect_trash_api(file: UploadFile = File(...), postoffice_id: str = Form(""), place: str = Form(""), time: str = Form("")):
    logger.info(f"Postoffice ID: {postoffice_id} Place: {place} Time: {time}")
    try:
        # Read the image file in memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(image)
        frame = funny(frame, postoffice_id, place, time)
        if frame is False:
            return JSONResponse(content={"status": False ,"message": "No trash detected."})

        # Convert the frame back to an image
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")

        return JSONResponse(content={"status": True, 'message': 'Trash detected.'})
    except Exception as e:
        logger.error(f"Error in detect_trash_api function: {e}")
        return JSONResponse(content={"status": False, "message": "An error occurred."})

##############################################################################################
##############################################################################################
#command to run this file
# uvicorn main:app --host