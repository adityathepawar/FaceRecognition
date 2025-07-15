# event_logger.py

from pymongo import MongoClient
from datetime import datetime
import gridfs
import cv2

# MongoDB connection
client = MongoClient("mongodb+srv://adityapawar2:jaypee@encodings.udy73ch.mongodb.net/?retryWrites=true&w=majority&appName=encodings")
db = client["events"]
fs = gridfs.GridFS(db)
collection = db["unwanted_events"]

def log_event(username, event_type, image_frame):
    timestamp = datetime.utcnow()
    image_name = f"{username}_{event_type}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"

    # Encode image frame to bytes
    is_success, buffer = cv2.imencode(".jpg", image_frame)
    if not is_success:
        print("❌ Failed to encode image")
        return

    file_id = fs.put(buffer.tobytes(), filename=image_name)

    doc = {
        "username": username,
        "event_type": event_type,
        "timestamp": timestamp,
        "image": file_id,
        "image_name": image_name
    }

    collection.insert_one(doc)
    print(f"✅ Logged {event_type} for {username}")
