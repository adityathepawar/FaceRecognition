import streamlit as st
from pymongo import MongoClient
import gridfs
from PIL import Image
import io
from datetime import datetime, timedelta

# MongoDB setup
client = MongoClient("mongodb+srv://adityapawar2:jaypee@encodings.udy73ch.mongodb.net/?retryWrites=true&w=majority&appName=encodings")
db = client["events"]
fs = gridfs.GridFS(db)
collection = db["unwanted_events"]

# Streamlit UI
st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("🛡️ Admin Dashboard – Event Monitoring")

# ============================ Sidebar Filters ============================
st.sidebar.header("📂 Filter Options")
user_filter = st.sidebar.text_input("🔍 Filter by Username")
event_filter = st.sidebar.selectbox("🎯 Event Type", ["All", "unknown_face", "smart_device"])
limit = st.sidebar.slider("📄 Number of Events", 1, 100, 20)

# Date range filter (UTC time assumed)
st.sidebar.markdown("📅 **Date & Time Filter**")

default_start = datetime.utcnow() - timedelta(days=7)
default_end = datetime.utcnow()

st.sidebar.markdown("📅 **Date & Time Filter**")

start_date = st.sidebar.date_input("Start Date", value=default_start.date())
start_time = st.sidebar.time_input("Start Time", value=default_start.time())

end_date = st.sidebar.date_input("End Date", value=default_end.date())
end_time = st.sidebar.time_input("End Time", value=default_end.time())

# Combine into full datetime objects
start_datetime = datetime.combine(start_date, start_time)
end_datetime = datetime.combine(end_date, end_time)


# ============================ MongoDB Query ============================
query = {
    "timestamp": {
        "$gte": start_datetime,
        "$lte": end_datetime
    }
}

if user_filter:
    query["username"] = user_filter
if event_filter != "All":
    query["event_type"] = event_filter

# ============================ Fetch Events ============================
events = list(collection.find(query).sort("timestamp", -1).limit(limit))

# ============================ Display Events ============================
if not events:
    st.warning("🚫 No events found for the selected filters.")
else:
    st.markdown(f"### 🧾 {len(events)} Event(s) Found")
    for event in events:
        col1, col2 = st.columns([1, 3])
        with col1:
            try:
                image_data = fs.get(event["image"]).read()
                image = Image.open(io.BytesIO(image_data))
                st.image(image, width=250, caption=event["event_type"])
            except:
                st.error("❌ Could not load image.")

        with col2:
            st.markdown(f"**👤 Username:** `{event['username']}`")
            st.markdown(f"**🕒 Timestamp:** `{event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}`")
            st.markdown(f"**📸 Image Name:** `{event['image_name']}`")
            st.markdown("---")
