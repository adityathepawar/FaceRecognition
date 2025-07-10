# [YOUR EXISTING IMPORTS]
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os
from ultralytics import YOLO
from datetime import datetime
from pymongo import MongoClient

# setting up mongoDB
client = MongoClient("mongodb+srv://adityapawar2:jaypee@encodings.udy73ch.mongodb.net/?retryWrites=true&w=majority&appName=encodings")

db = client["face_encodings"]
collection = db["face_recognition"]

# Setup InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

# Setup YOLOv8 Model
yolo_model = YOLO('yolov8n.pt')

st.set_page_config(page_title="Face Registration & Recognition", layout="centered")
st.title("🧠 Face Registration, Recognition & Smart Device Detection")

ordered_poses = ["Looking Front", "Looking Left", "Looking Right", "Looking Up", "Looking Down"]
captured_encodings = {pose: None for pose in ordered_poses}
front_face_image = None
current_pose_index = 0

user_name = st.text_input("Enter your name for registration:")
register = st.checkbox("📸 Start Registration")
recognize = st.checkbox("🕵️ Live Recognition")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)
mp_drawing = mp.solutions.drawing_utils

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 70:
        return "Too Dark"
    elif brightness > 180:
        return "Too Bright"
    else:
        return "Good"

def save_event_image(frame, employee_name):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    emp_folder = os.path.join("images", employee_name, "events")
    os.makedirs(emp_folder, exist_ok=True)
    filepath = os.path.join(emp_folder, f"{timestamp}.jpg")
    cv2.imwrite(filepath, frame)

def get_pose_direction(landmarks, image_shape):
    image_h, image_w = image_shape
    key_indices = [1, 33, 263, 61, 291, 199]
    pts = [landmarks[i] for i in key_indices]
    image_points = np.array([(pt.x * image_w, pt.y * image_h) for pt in pts], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (-30.0, -30.0, -30.0),
        (30.0, -30.0, -30.0),
        (-30.0, 30.0, -30.0),
        (30.0, 30.0, -30.0),
        (0.0, 50.0, 0.0)
    ])

    focal_length = image_w
    center = (image_w / 2, image_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        return "Undetected", None, None

    rvec_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]

    if pitch > 15:
        return "Looking Up", pitch, yaw
    elif pitch < -15:
        return "Looking Down", pitch, yaw
    elif yaw < -15:
        return "Looking Right", pitch, yaw
    elif yaw > 15:
        return "Looking Left", pitch, yaw
    else:
        return "Looking Front", pitch, yaw

def extract_encoding(frame):
    faces = app.get(frame)
    if faces:
        return faces[0].embedding.tolist()
    return None

FRAME_WINDOW = st.image([])

# ========== REGISTRATION ==========
if register:
    if not user_name:
        st.warning("⚠️ Enter your name first.")
    elif collection.find_one({"name": user_name.strip()}):
        st.error(f"❌ User '{user_name}' already exists.")
    else:
        cap = cv2.VideoCapture(0)
        st.info("📸 Capturing 5 face angles...")

        while register and current_pose_index < len(ordered_poses):
            ret, frame = cap.read()
            if not ret:
                break

            lighting_status = check_lighting(frame)
            cv2.putText(frame, f"Lighting: {lighting_status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if lighting_status != "Good":
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            expected_pose = ordered_poses[current_pose_index]
            detected_pose = "Face Not Detected"

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    detected_pose, pitch, yaw = get_pose_direction(landmarks.landmark, frame.shape[:2])

                    if detected_pose == expected_pose and captured_encodings[detected_pose] is None:
                        encoding = extract_encoding(rgb_frame)
                        if encoding:
                            captured_encodings[detected_pose] = encoding
                            if detected_pose == "Looking Front":
                                front_face_image = frame.copy()
                            current_pose_index += 1

                    mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            cv2.putText(frame, f"Required: {expected_pose}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Detected: {detected_pose}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        if all(v is not None for v in captured_encodings.values()):
            mongo_entry = {
                "name": user_name.strip(),
                "encodings": {pose: captured_encodings[pose] for pose in ordered_poses}
            }
            collection.insert_one(mongo_entry)
            st.success(f"✅ All 5 poses saved for {user_name.strip()}")

            user_folder = os.path.join("images", user_name.strip())
            os.makedirs(os.path.join(user_folder, "events"), exist_ok=True)
            front_path = os.path.join(user_folder, "front.jpg")
            if front_face_image is not None:
                cv2.imwrite(front_path, front_face_image)

# ========== RECOGNITION ==========
if recognize:
    documents = list(collection.find())
    if not documents:
        st.warning("⚠️ No registered users found.")
    else:
        known_encodings = []
        known_names = []
        for doc in documents:
            for pose, encoding in doc["encodings"].items():
                known_encodings.append(encoding)
                known_names.append(doc["name"])

        cap = cv2.VideoCapture(0)
        st.info("🕵️ Starting Multi-Face Recognition + Smart Device Detection...")

        pitch_yaw_history = {}

        while recognize:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            detected_faces = app.get(rgb_frame)

            device_detected = False
            yolo_results = yolo_model(frame, verbose=False)[0]
            for result in yolo_results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls_id = map(int, result[:6])
                label = yolo_model.names[cls_id]
                if label.lower() in ['cell phone', 'laptop', 'tv', 'remote', 'keyboard']:
                    device_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 215, 0), 2)
                    cv2.putText(frame, f"{label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)

            for face in detected_faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                encoding = face.embedding
                similarities = cosine_similarity([encoding], known_encodings)[0]
                best_idx = np.argmax(similarities)
                best_match = known_names[best_idx]
                similarity = similarities[best_idx]

                if similarity > 0.5:
                    employee_name = best_match
                else:
                    employee_name = "unknown"

                if employee_name == "unknown" or device_detected:
                    save_event_image(frame, employee_name)

                liveness_label = "Unknown"
                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        pose, pitch, yaw = get_pose_direction(landmarks.landmark, frame.shape[:2])
                        box_id = f"{x1}-{y1}-{x2}-{y2}"
                        if box_id in pitch_yaw_history:
                            old_pitch, old_yaw = pitch_yaw_history[box_id]
                            if abs(pitch - old_pitch) > 5 or abs(yaw - old_yaw) > 5:
                                liveness_label = "Live"
                            else:
                                liveness_label = "⚠️ Spoof"
                        else:
                            liveness_label = "Live"
                        pitch_yaw_history[box_id] = (pitch, yaw)
                        break

                color = (0, 255, 0) if liveness_label == "Live" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{employee_name} | {liveness_label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
