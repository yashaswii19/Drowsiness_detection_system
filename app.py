import streamlit as st
import cv2
import mediapipe as mp
from scipy.spatial import distance
import winsound
import time

# 🎨 PAGE CONFIG → sets page title and layout
st.set_page_config(page_title="Drowsiness Detection", layout="wide")

# 🌟 HEADER → main title and subtitle
st.markdown("""
    <h1 style='text-align: center; color: #00FFFF;'>🚗 Driver Drowsiness Detection System</h1>
    <p style='text-align: center; color: gray;'>Real-time fatigue monitoring using Computer Vision</p>
""", unsafe_allow_html=True)

# 📌 SIDEBAR → control panel for user interaction
st.sidebar.title("⚙️ Control Panel")

# Start and Stop buttons
start = st.sidebar.button("▶ Start Detection")
stop = st.sidebar.button("⏹ Stop Detection")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.write("👩‍💻 Student Project")
st.sidebar.write("📌 OpenCV + MediaPipe")
st.sidebar.write("🧠 EAR-based detection")

# ✅ SESSION STATE → maintains whether app is running
if "running" not in st.session_state:
    st.session_state.running = False

# Update running state based on button clicks
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# 🎯 MAIN LAYOUT → split screen into video + status panel
col1, col2 = st.columns([3, 1])

# Placeholder for webcam frame
frame_placeholder = col1.empty()

# Right-side status panel
with col2:
    st.subheader("📊 Status Panel")
    status_text = st.empty()  # shows drowsiness status
    info_text = st.empty()    # shows EAR value

# 🎥 MEDIAPIPE SETUP → initialize face mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# 👁️ Eye landmark indices (MediaPipe points)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 👁️ FUNCTION: Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points, landmarks, w, h):
    """
    Calculates EAR using 6 eye landmarks.
    EAR helps determine if the eye is open or closed.
    """

    points = []

    # Convert normalized landmark points to pixel coordinates
    for point in eye_points:
        x = int(landmarks[point].x * w)
        y = int(landmarks[point].y * h)
        points.append((x, y))

    # Vertical distances
    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])

    # Horizontal distance
    C = distance.euclidean(points[0], points[3])

    # EAR formula
    return (A + B) / (2.0 * C)

# 🚀 MAIN LOGIC → runs when detection is started
if st.session_state.running:

    # Start webcam
    cap = cv2.VideoCapture(0)

    # Timer to track eye closure duration
    start_time = None

    while cap.isOpened() and st.session_state.running:

        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            status_text.error("Camera not working")
            break

        # Get frame dimensions
        h, w = frame.shape[:2]

        # Convert BGR to RGB (required for MediaPipe)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect facial landmarks
        result = face_mesh.process(rgb)

        # If face is detected
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:

                landmarks = face_landmarks.landmark

                # Calculate EAR for both eyes
                left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
                right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)

                # Average EAR
                ear = (left_ear + right_ear) / 2

                # Display EAR value on UI
                info_text.info(f"EAR Value: {ear:.2f}")

                # 🧠 DROWSINESS LOGIC
                if ear < 0.25:

                    # Start timer when eyes close
                    if start_time is None:
                        start_time = time.time()

                    # Calculate duration of eye closure
                    elapsed = time.time() - start_time

                    # If eyes closed for 5 seconds → drowsy
                    if elapsed >= 5:
                        status_text.error("🚨 DROWSY DETECTED")
                        winsound.Beep(1500, 800)
                    else:
                        status_text.warning(f"⚠ Closing Eyes ({int(5 - elapsed)}s)")

                else:
                    # Reset timer if eyes open
                    start_time = None
                    status_text.success("😊 AWAKE")

        # Display video frame in UI
        frame_placeholder.image(frame, channels="BGR")

    # Release webcam after stopping
    cap.release()

else:
    # Message when system is not running
    st.info("Click 'Start Detection' from the sidebar to begin")

# 🔚 FOOTER
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using Machine Learning</p>", unsafe_allow_html=True)