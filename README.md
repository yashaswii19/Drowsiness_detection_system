# 🚗 Driver Drowsiness Detection System

This project detects whether a driver is **drowsy or awake** using a webcam.
It uses eye detection and alerts the user if eyes remain closed for a long time.

---

## 📌 Features

* Real-time webcam detection
* Detects eye closure
* Gives alert when driver is sleepy
* Simple UI using Streamlit

---

## 🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe
* Streamlit

---

## ⚙️ How It Works

1. Webcam captures video
2. Face and eyes are detected using MediaPipe
3. Eye Aspect Ratio (EAR) is calculated
4. If eyes are closed for 5 seconds → Drowsy alert is shown

---

## ▶️ How to Run

Install libraries:

```id="vsh7wl"
pip install opencv-python mediapipe streamlit scipy
```

Run the project:

```id="5nf4pc"
streamlit run app.py
```

---

## 📊 Output

* 😊 AWAKE
* ⚠ Closing Eyes
* 🚨 DROWSY

---

## 🧠 Algorithm

* MediaPipe Face Mesh
* Eye Aspect Ratio (EAR)

---


