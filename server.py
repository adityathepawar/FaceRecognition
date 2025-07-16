from fastapi import FastAPI, Request
import base64
import os

app = FastAPI()

@app.post("/event")
async def receive_event(request: Request):
    data = await request.json()
    username = data.get("username")
    event_type = data.get("event_type")
    timestamp = data.get("timestamp")
    image_base64 = data.get("image_base64")

    if not all([username, event_type, image_base64]):
        return {"error": "Missing data"}, 400

    # Save image
    folder = f"events/{username}"
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{event_type}_{timestamp.replace(':', '-')}.jpg"
    with open(filename, "wb") as f:
        f.write(base64.b64decode(image_base64))

    return {"message": "✅ Event received"}
