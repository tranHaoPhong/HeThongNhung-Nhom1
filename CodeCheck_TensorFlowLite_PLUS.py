import numpy as np
import cv2
import autopy
import mediapipe as mp
import urllib.request
import tensorflow as tf
import requests

url = "http://192.168.4.1/cam-hi.jpg"
# Địa chỉ IP của ESP32-CAM
esp32cam_ip = "192.168.4.1"

wCam, hCam = 800, 600
frameR = 100
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

wScr, hScr = autopy.screen.size()

# Load mô hình đã lưu (TensorFlow Lite)
interpreter = tf.lite.Interpreter(model_path="MODEL/hand_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    landmarks_data = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                image_height, image_width, _ = img.shape
                x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                landmarks_data += f"{x},{y},"
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks_data = landmarks_data[:-1]
    
    # Đảm bảo rằng landmarks_data không rỗng trước khi dự đoán
    if landmarks_data:
        # Chuyển landmarks_data thành mảng numpy 2D
        landmarks = np.array([list(map(int, point.split(','))) for point in landmarks_data.split(',')])

        # Chuẩn bị dữ liệu đầu vào cho mô hình TensorFlow Lite
        input_data = np.array(landmarks.reshape(1, -1), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Thực thi mô hình TensorFlow Lite
        interpreter.invoke()

        # Lấy kết quả từ mô hình TensorFlow Lite
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(output_data)
        print("Success:")

        #Chuyển đổi predicted_label sang kiểu dữ liệu int
        predicted_label = int(predicted_label)

        # Gửi dữ liệu dự đoán lên ESP32-CAM thông qua yêu cầu POST
        post_url = f"http://{esp32cam_ip}/post-data"
        post_data = {
            "Predict": predicted_label
        }
        post_response = requests.post(post_url, json=post_data)

        # In ra phản hồi từ ESP32-CAM
        print(post_response.text)
    # Đọc ảnh nền
    imageNEN = cv2.imread('Logo_PTIT_University.png')
    cv2.imshow("Image", imageNEN)

    # cv2.imshow("Image", img)
    
    key = cv2.waitKey(1)
    if key == 27: # ASCII code for Esc key
        break

cv2.destroyAllWindows()
