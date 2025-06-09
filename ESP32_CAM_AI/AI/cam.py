import cv2
import numpy as np
import urllib.request
import mediapipe as mp
import threading
import pygame

# IP ESP32-CAM
url = 'http://192.168.137.116/capture'

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Khởi tạo pygame mixer
pygame.mixer.init()

# Danh sách điểm quanh mắt
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_points):
    def euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    A = euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
    B = euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
    C = euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
    return (A + B) / (2.0 * C)

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(r"D:\KY_8\TRUNG\T2_RTOS\CODE\alarm.wav")
        pygame.mixer.music.play()

print("🚗 Đang chạy hệ thống phát hiện ngủ gật...")

cv2.namedWindow("ESP32-CAM - Giam sat giac ngu", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ESP32-CAM - Giam sat giac ngu", 960, 720)

while True:
    try:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status_text = "Normal"
        status_color = (0, 255, 0)
        color = (0, 255, 0)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < 0.22:
                    color = (0, 0, 255)
                    status_text = "Warning"
                    status_color = (0, 0, 255)
                    # 👉 Phát cảnh báo khi trạng thái là "Warning"
                    threading.Thread(target=play_alarm, daemon=True).start()
                else:
                    color = (0, 255, 0)
                    status_text = "Normal"
                    status_color = (0, 255, 0)

                for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    cx, cy = landmarks[idx]
                    cv2.circle(frame, (cx, cy), 2, color, -1)

        cv2.putText(frame, f"Status: {status_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("ESP32-CAM - Giam sat giac ngu", frame)
        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print("❌ Lỗi kết nối ESP32:", e)
        break

cv2.destroyAllWindows()
