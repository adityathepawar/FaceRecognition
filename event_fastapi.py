import requests
import base64
from datetime import datetime
import pytz

def log_event_api(username, event_type, image_frame):
    # Encode image to base64
    _, buffer = cv2.imencode(".jpg", image_frame)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    # IST timestamp
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).isoformat()

    # Prepare payload
    payload = {
        "username": username,
        "event_type": event_type,
        "timestamp": timestamp,
        "image_base64": image_base64
    }

    # Send POST request to your API endpoint
    try:
        response = requests.post("https://your-api-server.com/event", json=payload)
        if response.status_code == 200:
            print("✅ Event sent successfully")
        else:
            print("❌ Failed to send event:", response.status_code, response.text)
    except Exception as e:
        print("❌ Exception while sending event:", e)
