import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# Danh sách ký hiệu bạn huấn luyện
labels = ["K", "L", "M", "N", "O"]

# Cấu hình MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Hàm cắt vùng bàn tay + vẽ khung box

def extract_hand(image):
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            points = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark]
            x_min, y_min = max(min(p[0] for p in points) - 20, 0), max(min(p[1] for p in points) - 20, 0)
            x_max, y_max = min(max(p[0] for p in points) + 20, image.shape[1]), min(max(p[1] for p in points) + 20, image.shape[0])
            hand_crop = image[y_min:y_max, x_min:x_max]

            # Vẽ khung chữ nhật quanh tay để debug
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            if hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
                return hand_crop, image
    return None, image

# Hàm dự đoán

def detect_and_predict(frame, model, input_size):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_crop, annotated_frame = extract_hand(frame_rgb)
    if hand_crop is not None and hand_crop.size != 0:
        hand_crop = cv2.resize(hand_crop, input_size)
        input_image = hand_crop.reshape(1, *input_size, 3) / 255.0
        prediction = model.predict(input_image)
        return labels[np.argmax(prediction)], hand_crop, annotated_frame
    return None, None, frame_rgb

# Giao diện Streamlit

st.set_page_config(page_title="ASL Detector", layout="centered")
st.title("🖐️ Nhận diện ký hiệu ASL bằng Webcam")

model = load_model("customcnn_model.h5"); input_size = (192, 192); st.write("Mô hình CNN")
#model = load_model("mobilenetv2_model.h5"); input_size = (224, 224); st.write("Mô hình MobileNetV2")
#model = load_model("resnet50_model.h5"); input_size = (224, 224); st.write("Mô hình Resnet50")

run = st.checkbox("Bắt đầu webcam")

col1, col2 = st.columns([2, 1])

with col1:
    FRAME_WINDOW = st.image([])

with col2:
    CROP_WINDOW = st.empty()
    log_area = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    log_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, cropped, annotated = detect_and_predict(frame, model, input_size)

        if label:
            cv2.putText(annotated, f'Prediction: {label}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            log_list.append(f"[{time.strftime('%H:%M:%S')}] -> {label}")
        

        FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))


        if cropped is not None:
            CROP_WINDOW.image(cropped, caption="Vùng tay đã crop")

        log_area.text("\n".join(log_list[-5:]))

        if not run:
            break

    cap.release()
else:
    st.write("👉 Bật webcam để bắt đầu.")
