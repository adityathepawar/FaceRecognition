from pymongo import MongoClient
from datetime import datetime
import pytz
import gridfs
import cv2

# MongoDB connection
client = MongoClient("mongodb+srv://adityapawar2:jaypee@encodings.udy73ch.mongodb.net/?retryWrites=true&w=majority&appName=encodings")
db = client["events"]
fs = gridfs.GridFS(db)
collection = db["unwanted_events"]

# IST timezone setup
IST = pytz.timezone('Asia/Kolkata')

def log_event(username, event_type, image_frame):
    # Get IST timestamp
    utc_now = datetime.utcnow()
    ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(IST)

    # Format filename with IST timestamp
    image_name = f"{username}_{event_type}_{ist_now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"

    # Encode image frame to bytes
    is_success, buffer = cv2.imencode(".jpg", image_frame)
    if not is_success:
        print("❌ Failed to encode image")
        return

    # Save image to GridFS
    file_id = fs.put(buffer.tobytes(), filename=image_name)

    # Log document in MongoDB
    doc = {
        "username": username,
        "event_type": event_type,
        "timestamp": ist_now,
        "image": file_id,
        "image_name": image_name
    }

    collection.insert_one(doc)
    print(f"✅ Logged {event_type} for {username} at {ist_now}")
