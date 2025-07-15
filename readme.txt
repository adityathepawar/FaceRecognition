==============================
Face Pose Registration System
==============================

📌 ABOUT THE PROJECT
---------------------
This project is a real-time face registration and pose validation system built using Python and Streamlit. It captures a user's face from four key angles—front-facing, left profile, right profile, and looking down—using a webcam interface. Each pose is validated using facial landmarks detected via the InsightFace `buffalo_l` model to ensure accuracy. Once all poses are captured, a 128-dimensional face embedding is computed and stored along with the user's name in an Excel file or MongoDB database. The system includes live face tracing, pose prompts, and real-time feedback. This solution is ideal for employee identity verification and attendance monitoring.

🛠️ SETUP INSTRUCTIONS
-----------------------
1. Install Python 3.8 (recommended).
2. Create and activate a virtual environment.
3. https://visualstudio.microsoft.com/visual-cpp-build-tools/
   Visit the link for downloading visual studio build tools and inside it download desktop tools for c++. (it's for installing insigtface library as it build on c++)
4. Install required libraries:
   pip install streamlit opencv-python insightface pandas numpy openpyxl
5. Download the `buffalo_l` model from:
   https://huggingface.co/deepinsight/insightface/tree/main/models/buffalo_l
6. Place the model in:
   C:\Users\<YourUsername>\.insightface\models\buffalo_l
   (Maintain the full folder structure as shown above.)

▶️ USAGE
---------
1. Open your terminal or command prompt.
2. Navigate to the folder where your script is located.
3. Run the application using:
   streamlit run your_script_name.py
4. Enter your name in the app and follow on-screen instructions to capture face poses.
5. After successful capture and encoding, your data will be saved automatically.

✅ OUTPUT
----------
- Encoded face vectors are saved in `encodings.xlsx` or a MongoDB collection.
- Captured images are stored in the `known_faces/` directory (optional).
- Facial pose validation ensures only high-quality angle data is accepted.

-----------------------
Built with Python, Streamlit, OpenCV, and InsightFace.

1. python version 3.8
2. visual studio build tools - (inside the application install c++ build tools)
3. libraries :- streamlit pymongo gridfs numpy pillow scikit-learn opencv-python mediapipe    insightface ultralytics onnxruntime
4. A mongoDb server
