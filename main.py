import cv2
import mediapipe as mp
from scipy.spatial import distance
import winsound
import time   # NEW

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_points, landmarks, w, h):
    points = []
    for point in eye_points:
        x = int(landmarks[point].x * w)
        y = int(landmarks[point].y * h)
        points.append((x, y))

    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])

    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.25
start_time = None   # NEW

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)

            ear = (left_ear + right_ear) / 2

            # 🔥 NEW LOGIC
            if ear < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()

                elapsed = time.time() - start_time

                if elapsed >= 5:   # 5 seconds delay
                    cv2.putText(frame, "DROWSY!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    winsound.Beep(1500, 800)
                else:
                    cv2.putText(frame, "Closing Eyes...", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                start_time = None
                cv2.putText(frame, "AWAKE", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()