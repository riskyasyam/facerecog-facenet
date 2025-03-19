# Code berikut untuk mengambil sebuah foto yang auto crop ke muka 
# Sehnigga sudah tidak perlu pre processing data lagi. 
# Namun akan saya berikan ode untuk pre processing datanya pada file lain jika dataset bukan diambil dari code disini

import cv2
import os
import time  # Tambahkan library time

# Buka kamera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists("dataset_asyam"):
    os.makedirs("dataset_asyam")

count = 0
while count < 50:  # Ambil 50 wajah
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))  # Resize sesuai kebutuhan FaceNet
        cv2.imwrite(f'dataset_asyam/face_{count}.jpg', face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print(f"Face {count} captured.")
        
        time.sleep(1)  # Jeda 1 detik untuk ganti pose

    cv2.imshow('Face Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
